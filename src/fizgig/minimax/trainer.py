"""MiniMax H3 — image-only training core: flow-matching loss + timestep sampling.

The heart of the trainer, isolated so it's headless-testable with the tiny model (no GPU,
no 66 GB base, no 32 B text encoder). The full LoRA/rotating-FT wiring, caching and GUI come
later; this pins the maths of one training step.

Flow / sign convention (matched to ComfyUI's comfy/ldm/minimax/model.py):
  x0 = clean latent, noise ~ N(0,1), sigma in (0,1) the noise level.
  noised = (1 - sigma)*x0 + sigma*noise            (sigma 0 = clean, 1 = pure noise)
  t = 1 - sigma                                     the "cleanness" fed to the time embedder
  the DiT's raw video_out predicts (x0 - noise)     (the reference NEGATES it to get the
                                                     sampler's velocity noise - x0)
So the training target for the model's output is `x0 - noise`.
"""

import argparse
import logging
import math
import os
import sys
import time
from multiprocessing import Value

import torch
import torch.nn.functional as F

from fizgig.training.metadata import ARCHITECTURE_MINIMAX

logger = logging.getLogger(__name__)

# LoRA targets the transformer blocks' ATTENTION + MLP Linears only (+ the 2-block text
# refiner). The fp32 patch/head IO layers are left alone (wrapping them clashes fp32-base vs
# bf16-adapter) — and `adaln_proj.linear` is deliberately EXCLUDED:
#   * ComfyUI's pruned inference builds (minimax_h3_*_pruned_*) replace the full-width AdaLN
#     with a time-embedding curve table — adaln_proj.linear there is [96768, 8], not
#     [96768, 2688], so an adaln LoRA can never apply (one shape error per block, key dropped).
#   * adaln up-matrices are 96768-out (6x qkv) — training them poured the largest share of LoRA
#     capacity into weights the user's inference model then threw away (real run: ~50% likeness
#     until excluded).
# Power users on the full bf16 model can re-add it via --include_patterns.
DEFAULT_INCLUDE_PATTERNS = [r"blocks\.\d+\.attn\..*", r"blocks\.\d+\.mlp\..*",
                            r"token_refiner\.blocks\..*"]


# ---------------------------------------------------------------------------
# VRAM planner — resolves "auto" block swap + gradient checkpointing from the card's actual
# free VRAM and the run's real token load (bucket megapixels x batch). Simpler than Krea 2's:
# one quant mode (NF4), batch is 1, no preview co-residency.
# ---------------------------------------------------------------------------
# Measured anchors (5090, real 33B, rank 16, ~0.2 MP batch 1 — GPU validation pass, 4 Aug):
#   no swap, no ckpt : resident 17.6, step peak 22.7  (overhead ~5.1)
#   no swap, ckpt    : resident 17.5, step peak 18.3  (overhead 0.9 — and only ~+0.1 s/step)
#   swap 16 + ckpt   : resident 11.9 (0.34 GB/block), steady 12.8, step peak 19.3 — the swap
#                      path carries a ~7.4 GB backward transient (checkpoint recompute segments
#                      held by the engine), which the planner must budget on top of residency.
_RESIDENT_GB = 17.5          # full NF4 base resident (measured 17.3-17.6)
_PER_BLOCK_GB = 0.34         # one parked block's GPU share (measured: (17.5-11.9)/16)
_ACT_GB_NOCKPT = 5.5         # step overhead at 0.25 MP batch 1, no checkpointing (measured 5.1)
_ACT_GB_CKPT = 2.0           # step overhead at 0.25 MP batch 1, checkpointed (measured 0.9; margin)
_SWAP_TRANSIENT_GB = 7.5     # extra backward-time peak whenever swap is active (measured 7.4 @ n=16)
_RESERVE_GB = 1.5            # display / allocator / fragmentation headroom


def plan_vram(free_gb: float, mp: float = 0.25, batch: int = 1):
    """Pure planner: (blocks_to_swap, gradient_checkpointing) from free VRAM + token load.

    Token load scales the activation term linearly (tokens ∝ mp x batch). Checkpointing is
    preferred OFF (faster) when everything fits without it; forced ON whenever swap is needed
    (without recompute, autograd would pin every swapped block's weights through backward).
    Swap additionally budgets _SWAP_TRANSIENT_GB: the backward pass transiently holds
    recompute segments beyond the parked residency (measured, see anchors above)."""
    scale = max(0.25, float(mp)) / 0.25 * max(1, int(batch))
    need_nockpt = _RESIDENT_GB + _ACT_GB_NOCKPT * scale + _RESERVE_GB
    if free_gb >= need_nockpt:
        return 0, False
    need_ckpt = _RESIDENT_GB + _ACT_GB_CKPT * scale + _RESERVE_GB
    if free_gb >= need_ckpt:
        return 0, True
    deficit = need_ckpt + _SWAP_TRANSIENT_GB - free_gb
    blocks = min(40, int(deficit / _PER_BLOCK_GB + 0.999))
    return blocks, True


def sample_sigmas(batch: int, device, shift: float = None, generator=None,
                  image_tokens: int = None) -> torch.Tensor:
    """Noise levels in (0,1) for training.

    shift=None (the default): the validated IMAGE recipe — logit-normal base
    (sigmoid(randn), the SD3/Flux/Krea 2 density) with a resolution-dependent shift
    (shift = exp(mu), mu linear in the image-token count: 256 tokens -> 0.5, 6400 -> 1.15 —
    the same mapping Krea 2 trains with). At 0.25 MP (~225 tokens) that's shift ~1.65:
    median sigma ~0.62 with real coverage of the low-sigma zone where identity detail lives.

    shift=<float>: legacy uniform-u + shift map (sigma = s*u/(1+(s-1)*u)). NOTE: H3's
    sigma_shift_video=12.0 is the SAMPLER'S step-spacing schedule for full VIDEO token counts —
    training image LoRAs with it put 57% of steps at sigma>0.9 and only ~4% below 0.3, which
    produced structurally poor likeness at every checkpoint (real-run finding). Left available
    for experiments only.
    """
    if shift is None:
        tokens = float(image_tokens or 225)                       # ~0.25 MP default
        mu = 0.5 + (tokens - 256.0) * (1.15 - 0.5) / (6400.0 - 256.0)
        shift = math.exp(mu)
        base = torch.sigmoid(torch.randn(batch, device=device, generator=generator))
    else:
        base = torch.rand(batch, device=device, generator=generator)
    return (shift * base) / (1.0 + (shift - 1.0) * base)


def compute_loss(model, latent: torch.Tensor, text_embeds: torch.Tensor, *,
                 sigma: torch.Tensor = None, shift: float = None, generator=None,
                 noise: torch.Tensor = None):
    """One image-training step's loss.

    latent      : [1, 24, 1, H, W] clean VAE latent (x0).
    text_embeds : [1, L, text_dim] Qwen3-VL states.
    noise       : optional fixed noise (reproducible steps / tests); else sampled.
    Returns (loss, sigma_used) — MSE of the DiT's video_out against (x0 - noise).
    """
    if latent.shape[0] != 1:
        raise ValueError("MiniMax H3 image training is batch size 1")
    device = latent.device
    x0 = latent.float()
    # The DiT patchifies with patch_size (1, ph, pw), so the latent's H and W must be divisible by
    # the spatial patch. The dataset buckets on a 16-px step and the VAE is 16x, so a latent can be
    # odd (e.g. a 496-px bucket -> 31-px latent, not divisible by 2). Crop to the patch multiple
    # (drops at most one latent row/col = <=16 px of image edge) so patchify is exact and the target
    # (x0 - noise) stays the same shape as the model's prediction.
    _pt, _ph, _pw = getattr(model, "patch_size", (1, 2, 2))
    _H, _W = x0.shape[-2], x0.shape[-1]
    _Hc, _Wc = (_H // _ph) * _ph, (_W // _pw) * _pw
    if (_Hc, _Wc) != (_H, _W):
        x0 = x0[..., :_Hc, :_Wc].contiguous()
    if noise is None:
        noise = torch.randn(x0.shape, device=device, generator=generator, dtype=torch.float32)
    else:
        noise = noise.to(device=device, dtype=torch.float32)[..., :x0.shape[-2], :x0.shape[-1]]
    if sigma is None:
        # Resolution-aware auto schedule: token count from the (cropped) latent's patch grid.
        _tokens = (x0.shape[-2] // _ph) * (x0.shape[-1] // _pw)
        sigma = sample_sigmas(1, device, shift=shift, generator=generator, image_tokens=_tokens)
    s = sigma.reshape(1, 1, 1, 1, 1).to(torch.float32)

    noised = (1.0 - s) * x0 + s * noise
    t = (1.0 - sigma).to(device)
    pred = model(noised.to(latent.dtype), t, text_embeds)
    target = (x0 - noise).to(pred.dtype)
    return F.mse_loss(pred.float(), target.float()), float(sigma.reshape(-1)[0])


# ---------------------------------------------------------------------------
# Adaptive LR — bi-directional plateau tracker (architecture-agnostic; a faithful port of the
# Klein/Krea 2 watcher). Stability signal is weight-norm growth (>30%), same as Krea 2 (the H3
# loop clips gradients but the watcher reads weight-norm growth, not the clip ratio).
# ---------------------------------------------------------------------------
class AdaptiveLR:
    """Each epoch boundary: probe UP x1.25 on steady loss descent (patience 2); reduce DOWN x0.5
    on loss plateau (patience ramp) or a stability signal. On a stability event it blends the LoRA
    weights 70/30 toward the previous epoch's snapshot and restores the optimizer state (kills bad
    Adam momentum). The CPU rollback snapshot is in-memory only; the streak/best_loss scalars are
    JSON round-trippable (kept for parity — this barebones trainer has no resume yet)."""

    BLEND = 0.7
    WEIGHT_GROWTH_THRESHOLD = 0.30

    def __init__(self, min_lr, max_lr):
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.best_loss = None
        self.good_streak = 0
        self.bad_streak = 0
        self.stability_streak = 0
        self.stability_triggered = False
        self.prev_weight_norm = None
        self.snapshot = None  # {"weights": {...cpu...}, "optim": cpu state} — not persisted

    def state_dict(self):
        return {"best_loss": self.best_loss, "good_streak": self.good_streak,
                "bad_streak": self.bad_streak, "stability_streak": self.stability_streak,
                "stability_triggered": self.stability_triggered,
                "prev_weight_norm": self.prev_weight_norm}

    def load_state_dict(self, d):
        if not d:
            return
        self.best_loss = d.get("best_loss")
        self.good_streak = int(d.get("good_streak", 0))
        self.bad_streak = int(d.get("bad_streak", 0))
        self.stability_streak = int(d.get("stability_streak", 0))
        self.stability_triggered = bool(d.get("stability_triggered", False))
        self.prev_weight_norm = d.get("prev_weight_norm")

    @staticmethod
    def _weight_norm(network):
        wn = 0.0
        with torch.no_grad():
            for p in network.parameters():
                if p.requires_grad:
                    wn += float(p.detach().float().norm().item()) ** 2
        return wn ** 0.5

    def _snapshot(self, network, optimizer):
        with torch.no_grad():
            weights = {n: p.detach().clone().to("cpu")
                       for n, p in network.named_parameters() if p.requires_grad}

        def _cpu(o):
            if isinstance(o, torch.Tensor):
                return o.detach().clone().to("cpu")
            if isinstance(o, dict):
                return {k: _cpu(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_cpu(v) for v in o]
            return o
        try:
            self.snapshot = {"weights": weights, "optim": _cpu(optimizer.state_dict())}
        except Exception:
            self.snapshot = {"weights": weights, "optim": None}

    def _rollback(self, network, optimizer):
        cur = dict(network.named_parameters())
        with torch.no_grad():
            for name, prev in self.snapshot["weights"].items():
                if name in cur and cur[name].requires_grad:
                    p = cur[name]
                    prev_d = prev.to(device=p.device, dtype=p.dtype)
                    p.copy_(self.BLEND * prev_d + (1.0 - self.BLEND) * p)
        if self.snapshot.get("optim") is not None:
            try:
                optimizer.load_state_dict(self.snapshot["optim"])
            except Exception:
                pass

    def epoch_boundary(self, epoch, current_loss, network, optimizer):
        """epoch is 0-indexed (global). epoch 0 arms the baseline; epoch >= 1 adjusts the LR."""
        if epoch == 0:
            self.best_loss = current_loss
            self.prev_weight_norm = self._weight_norm(network)
            logger.info(f"[adaptive_lr] epoch 1: loss={current_loss:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} | ARMED")
            self._snapshot(network, optimizer)
            return

        patience_up = 2
        patience_down = 2 if (self.stability_triggered or epoch == 1 or epoch >= 4) else 1
        cur_lr = optimizer.param_groups[0]["lr"]
        new_lr = cur_lr
        cur_wn = self._weight_norm(network)
        weight_growth = None
        if self.prev_weight_norm and self.prev_weight_norm > 0:
            weight_growth = (cur_wn - self.prev_weight_norm) / self.prev_weight_norm
        stability_reason = None
        if weight_growth is not None and weight_growth > self.WEIGHT_GROWTH_THRESHOLD:
            stability_reason = f"wnorm_Δ {weight_growth*100:+.0f}% > {self.WEIGHT_GROWTH_THRESHOLD*100:.0f}%"

        action, reason = "HOLD", ""
        if stability_reason is not None:
            self.stability_streak += 1
            stability_patience = 1 if not self.stability_triggered else 2
            if self.stability_streak >= stability_patience:
                candidate = max(cur_lr * 0.5, self.min_lr)
                note = ""
                if self.snapshot is not None:
                    self._rollback(network, optimizer)
                    note = f"; blended {int(self.BLEND*100)}/{int((1-self.BLEND)*100)} + optim restored"
                if candidate < cur_lr:
                    new_lr = candidate
                    action = "REDUCE+ROLLBACK" if self.snapshot is not None else "REDUCE"
                else:
                    action = "HOLD (floored)"
                reason = f"stability: {stability_reason}{note}"
                self.good_streak = self.bad_streak = self.stability_streak = 0
                self.stability_triggered = True
            else:
                action = "WAIT"
                reason = f"stability: {stability_reason}, streak {self.stability_streak}/{stability_patience}"
        elif self.best_loss is None or current_loss < self.best_loss:
            self.stability_streak = 0
            self.best_loss = current_loss
            self.good_streak += 1
            self.bad_streak = 0
            if self.good_streak >= patience_up:
                candidate = min(cur_lr * 1.25, self.max_lr)
                if candidate > cur_lr:
                    new_lr = candidate
                    action = "PROBE UP"
                    reason = f"loss improving, streak {self.good_streak}"
                else:
                    action = "HOLD (capped)"
                    reason = "loss improving, at max_lr"
                self.good_streak = 0
            else:
                reason = f"loss improving, streak {self.good_streak}/{patience_up}"
        else:
            self.stability_streak = 0
            self.bad_streak += 1
            self.good_streak = 0
            if self.bad_streak >= patience_down:
                candidate = max(cur_lr * 0.5, self.min_lr)
                if candidate < cur_lr:
                    new_lr = candidate
                    action = "REDUCE"
                    reason = f"loss plateau, streak {self.bad_streak}"
                else:
                    action = "HOLD (floored)"
                    reason = "loss plateau, at min_lr"
                self.bad_streak = 0
            else:
                reason = f"loss plateau, streak {self.bad_streak}/{patience_down}"

        if new_lr != cur_lr:
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
        lr_str = f"{cur_lr:.2e}" if new_lr == cur_lr else f"{cur_lr:.2e}->{new_lr:.2e}"
        wn_str = f"{weight_growth*100:+.0f}%" if weight_growth is not None else "—"
        logger.info(f"[adaptive_lr] epoch {epoch + 1}: loss={current_loss:.4f} lr={lr_str} "
                    f"wnorm_Δ={wn_str} | {action} ({reason})")
        self.prev_weight_norm = cur_wn
        self._snapshot(network, optimizer)


# ---------------------------------------------------------------------------
# Full image-only training loop (NF4 base + LoRA) over the H3 caches.
# ---------------------------------------------------------------------------
class _Collator:
    """DataLoader batch_size is always 1 (the dataset batches internally by bucket)."""

    def __init__(self, shared_epoch, dataset):
        self.shared_epoch = shared_epoch
        self.dataset = dataset

    def __call__(self, examples):
        wi = torch.utils.data.get_worker_info()
        ds = wi.dataset if wi is not None else self.dataset
        ds.set_current_epoch(self.shared_epoch.value)
        return examples[0]


def _save_training_state(output_dir, output_name, network, optimizer, *, epoch, global_step,
                         dtype, extra=None):
    """Save a resumable training-state dir matching Klein/Krea 2 naming: <name>-<NNNNNN>-state/.

    NNNNNN is the number of COMPLETED epochs (= the next 0-indexed epoch to run). Holds the
    network weights in NATIVE state_dict naming (never the LyCORIS comfy-format rewrite — resume
    load_state_dict needs the module keys), the optimizer state, RNG, and a small JSON. The
    GUI's _detect_latest_state_dir finds the highest-numbered dir and passes it to --resume."""
    import json
    state_dir = os.path.join(output_dir, f"{output_name}-{epoch:06d}-state")
    os.makedirs(state_dir, exist_ok=True)
    network.save_weights(os.path.join(state_dir, "lora.safetensors"), dtype,
                         {"ss_architecture": ARCHITECTURE_MINIMAX,
                          "ss_network_module": "fizgig.minimax (state dir, native keys)"})
    torch.save(optimizer.state_dict(), os.path.join(state_dir, "optimizer.pt"))
    rng = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, os.path.join(state_dir, "rng.pt"))
    meta = {"epoch": epoch, "global_step": global_step}
    if extra:
        meta.update(extra)
    with open(os.path.join(state_dir, "training_state.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    logger.info(f"[state] saved -> {state_dir}")
    return state_dir


def _load_training_state(state_dir, network, optimizer, *, device):
    """Restore network + optimizer + RNG from a state dir. Returns (start_epoch, global_step, meta)."""
    import json
    from safetensors.torch import load_file
    # strict=False tolerates benign key drift, but if NOTHING matched the network silently stays
    # at its zero init and the run "succeeds" while training from scratch — then overwrites the
    # finished LoRA with a no-op. Refuse that outright.
    _incompat = network.load_state_dict(load_file(os.path.join(state_dir, "lora.safetensors")), strict=False)
    _missing = getattr(_incompat, "missing_keys", [])
    if _missing and len(_missing) >= len(network.state_dict()):
        raise RuntimeError(
            f"[state] {state_dir} matched none of this network's {len(network.state_dict())} keys — "
            f"refusing to resume into a zero-initialised network. The state was almost certainly "
            f"saved with a different config (rank/alpha/factor, network type, or target modules).")
    opt_path = os.path.join(state_dir, "optimizer.pt")
    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
    rng_path = os.path.join(state_dir, "rng.pt")
    if os.path.exists(rng_path):
        try:
            rng = torch.load(rng_path)
            torch.set_rng_state(rng["torch"].to("cpu", dtype=torch.uint8) if hasattr(rng["torch"], "to") else rng["torch"])
            if "cuda" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            logger.warning("[state] RNG restore failed; continuing with fresh RNG", exc_info=True)
    meta_path = os.path.join(state_dir, "training_state.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    return int(meta.get("epoch", 0)), int(meta.get("global_step", 0)), meta


def _save_lora(network, path, network_dim, network_alpha, dtype, extra_metadata=None):
    is_lokr = getattr(network, "_network_type", "lora") == "lokr"
    if is_lokr:
        metadata = {
            "ss_network_module": "fizgig.minimax (lokr, transformer blocks)",
            "ss_lokr_factor": str(getattr(network, "_lokr_factor", "")),
            "ss_architecture": ARCHITECTURE_MINIMAX,
        }
    else:
        metadata = {
            "ss_network_module": "fizgig.minimax (lora_unet, transformer blocks)",
            "ss_network_dim": str(network_dim),
            "ss_network_alpha": str(network_alpha),
            "ss_architecture": ARCHITECTURE_MINIMAX,
        }
    if extra_metadata:
        metadata.update(extra_metadata)
    if is_lokr:
        # LyCORIS-standard keys (diffusion_model.<dotted>.lokr_*) — the format every ComfyUI
        # LoKR in the wild uses. Unlike Krea 2 (whose internal saves stay native for resume and
        # previews), MiniMax has neither, and every checkpoint's only consumer is ComfyUI — so
        # every LoKR save is comfy-format. Fizgig's own loader ingests both namings via
        # ensure_kohya_lora_state_dict.
        from fizgig.networks.lora import _precalculate_safetensors_hashes
        from safetensors.torch import save_file
        dotted = getattr(network, "_dotted_names", {})
        sd = {}
        for k, v in network.state_dict().items():
            mod, _, suffix = k.partition(".")
            path_dotted = dotted.get(mod)
            nk = f"diffusion_model.{path_dotted}.{suffix}" if path_dotted else k
            v = v.detach().clone().to("cpu")
            if dtype is not None:
                v = v.to(dtype)
            sd[nk] = v
        model_hash, legacy_hash = _precalculate_safetensors_hashes(sd, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash
        save_file(sd, path, metadata)
        return
    network.save_weights(path, dtype, metadata)


def train_minimax(
    dataset_config: str,
    output_dir: str,
    output_name: str,
    dit_path: str,
    *,
    network_dim: int = 16,
    network_alpha: float = 16,
    network_type: str = "lora",      # "lora" | "lokr" (Kronecker, full-matrix w2)
    lokr_factor: int = 8,            # LoKR only: w1 is ~factor x factor; dim/alpha unused
    learning_rate: float = 1e-4,
    max_train_epochs: int = 10,
    save_every_n_epochs: int = 0,
    # Resumable state dirs (network + optimizer + RNG + adaptive scalars). Pause saves state
    # regardless of these — they govern only the automatic per-checkpoint / end-of-run saves.
    save_state: bool = False,
    save_state_on_train_end: bool = False,
    keep_last_n_states: int = 2,
    resume_state_dir: str = None,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    optimizer_type: str = "adamw8bit",
    optimizer_args: str = "",
    include_patterns: list = None,
    quantize: bool = True,           # NF4 the base (QLoRA); False = bf16 base (needs ~66 GB VRAM)
    shift: float = None,             # None = auto resolution schedule (logit-normal); float = legacy
    blocks_to_swap="auto",           # "auto" | int — park the last N blocks on CPU between uses
    gradient_checkpointing="auto",   # "auto" | "on" | "off" — forced on when swap > 0
    adaptive_lr: bool = False,
    adaptive_lr_min: float = 1e-5,
    adaptive_lr_max: float = 4e-4,
    # Output metadata (recorded in the saved LoRA).
    metadata_title: str = None,
    metadata_author: str = None,
    metadata_description: str = None,
    metadata_license: str = None,
    metadata_tags: str = None,
    metadata_trigger_phrase: str = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Native MiniMax H3 image-only LoRA training: bucketed dataloader over the H3 caches ->
    flow-matching loss -> optimizer -> save a ComfyUI-compatible LoRA. No samples, no preview."""
    from torch.utils.data import DataLoader

    from fizgig.dataset.config import (BlueprintGenerator, ConfigSanitizer,
                                       generate_dataset_group_by_blueprint, load_user_config)
    from fizgig.networks.lora import create_network
    from fizgig.training.optimizers import create_optimizer
    from fizgig.training.train_utils import LossRecorder
    from fizgig.training.metadata import build_metadata, resolve_title, ARCHITECTURE_MINIMAX
    from fizgig.minimax.loader import load_minimax_h3_dit
    from tqdm import tqdm
    import math

    torch.manual_seed(seed)
    include_patterns = include_patterns or DEFAULT_INCLUDE_PATTERNS

    # ---- dataset (built from the caches the two cache scripts wrote) ----
    shared_epoch = Value("i", 0)
    user_config = load_user_config(dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        user_config, argparse.Namespace(), architecture=ARCHITECTURE_MINIMAX)
    group = generate_dataset_group_by_blueprint(
        blueprint.dataset_group, training=True, num_timestep_buckets=None, shared_epoch=shared_epoch)
    if group.num_train_items == 0:
        raise RuntimeError("No training items — run minimax_cache_latents then minimax_cache_text first.")
    logger.info(f"MiniMax H3 training: {group.num_train_items} items, {max_train_epochs} epochs")
    if shift is None:
        logger.info("[timesteps] auto image schedule: logit-normal + resolution shift "
                    "(~1.65 @0.25MP — full coverage of the identity-detail zone)")
    else:
        logger.warning(f"[timesteps] explicit shift={shift} — uniform-u legacy map. Note: 12.0 is "
                       "the VIDEO sampler schedule and trains almost entirely at high noise; the "
                       "auto default (omit --shift) is right for image LoRAs.")

    # ---- VRAM plan: block swap + gradient checkpointing (before the base loads) ----
    _mp = 0.25
    try:
        _mp = max(w * h / 1e6 for ds in group.datasets for (w, h) in ds.batch_manager.bucket_resos)
    except Exception:
        pass
    _ckpt_req = str(gradient_checkpointing).lower()
    if str(blocks_to_swap).lower() == "auto":
        if torch.cuda.is_available() and quantize:
            _free_gb = torch.cuda.mem_get_info()[0] / 1e9
            n_swap, _ckpt_auto = plan_vram(_free_gb, mp=_mp)
            logger.info(f"[vram] auto plan: free {_free_gb:.1f} GB, largest bucket {_mp:.2f} MP "
                        f"-> blocks_to_swap={n_swap}, checkpointing={'on' if _ckpt_auto else 'off'}")
        else:
            n_swap, _ckpt_auto = 0, False
    else:
        n_swap = max(0, int(blocks_to_swap))
        _ckpt_auto = n_swap > 0
    use_ckpt = {"on": True, "off": False}.get(_ckpt_req, _ckpt_auto)
    if n_swap > 0 and not use_ckpt:
        logger.info("[vram] block swap needs gradient checkpointing (autograd would pin swapped "
                    "weights through backward) — forcing it on.")
        use_ckpt = True

    # ---- base (NF4-frozen) + trainable LoRA over the transformer blocks ----
    dit = load_minimax_h3_dit(dit_path, device=device, compute_dtype=dtype, quantize=quantize,
                              blocks_to_swap=n_swap)
    dit.requires_grad_(False)                                   # frozen base (QLoRA-style)
    if n_swap > 0:
        n_swap = dit.enable_block_swap(n_swap)                  # sets the JIT-move boundary
        logger.info(f"[vram] block swap active: last {n_swap} blocks parked on CPU "
                    f"(~{n_swap * 0.34:.1f} GB VRAM freed, packed NF4 in RAM)")
    if use_ckpt:
        dit.enable_gradient_checkpointing()
        logger.info("[vram] gradient checkpointing ON")
    if network_type == "lokr":
        # LoKR (Kronecker) — same mechanism as Krea 2's: module_class swaps the parametrization
        # inside the identical scan/wrap machinery, so include_patterns (adaln exclusion) and the
        # NF4/Linear4bit base compose unchanged. dim/alpha are ignored; factor is the dial.
        from fizgig.networks.lora import LoKRModule
        logger.info(f"network: LoKR (Kronecker), factor {lokr_factor}, full-matrix w2 — "
                    "dim/alpha do not apply")
        network = create_network(None, "lora_unet", 1.0, network_dim, network_alpha, None, [], dit,
                                 include_patterns=include_patterns,
                                 module_class=LoKRModule, module_kwargs={"factor": int(lokr_factor)})
    else:
        network = create_network(None, "lora_unet", 1.0, network_dim, network_alpha, None, [], dit,
                                 include_patterns=include_patterns)
    network.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
    network.requires_grad_(True)
    network.to(device=device, dtype=dtype)
    network._network_type = network_type
    network._lokr_factor = int(lokr_factor)
    # Dotted module paths for the LyCORIS-standard save (diffusion_model.<path>.lokr_*) — built
    # from the DiT itself with the same flattening create_modules used, so the reverse mapping is
    # exact even where module names contain underscores. isinstance covers bnb Linear4bit (an
    # nn.Linear subclass).
    network._dotted_names = {
        f"lora_unet_{name.replace('.', '_')}": name
        for name, m in dit.named_modules() if isinstance(m, torch.nn.Linear)
    }
    if network_type == "lokr":
        logger.info(f"LoKR: {len(network.unet_loras)} modules wrapped (factor {lokr_factor})")
    else:
        logger.info(f"LoRA: {len(network.unet_loras)} modules wrapped (dim {network_dim}, alpha {network_alpha})")

    params = list(network.get_trainable_params())

    # Adaptive LR ignores the Learning Rate box: it starts at the GEOMETRIC MIDPOINT of Min/Max
    # and the watcher owns the LR from there (matches Klein/Krea 2). Two knobs, not three.
    adaptive = AdaptiveLR(adaptive_lr_min, adaptive_lr_max) if adaptive_lr else None
    if adaptive:
        learning_rate = math.sqrt(adaptive_lr_min * adaptive_lr_max)
        logger.info(f"[adaptive_lr] ENABLED — start_lr={learning_rate:.3e} (geometric midpoint) "
                    f"min={adaptive_lr_min:.3e} max={adaptive_lr_max:.3e}; the Learning Rate box is ignored")

    optimizer, optimizer_label = create_optimizer(optimizer_type, params, learning_rate, optimizer_args)
    logger.info(f"optimizer: {optimizer_label} @ lr={learning_rate:.3e}")

    collator = _Collator(shared_epoch, group)
    loader = DataLoader(group, batch_size=1, shuffle=True, collate_fn=collator, num_workers=0)
    try:
        steps_per_epoch = len(loader)
    except TypeError:
        steps_per_epoch = group.num_train_items

    os.makedirs(output_dir, exist_ok=True)
    pause_flag = os.path.join(output_dir, ".pause_requested")

    # ---- resume: restore network + optimizer + RNG + (epoch, step) + adaptive scalars ----
    from fizgig.training.train_utils import prune_state_dirs
    global_step = 0
    start_epoch = 0
    if resume_state_dir and os.path.isdir(resume_state_dir):
        start_epoch, global_step, _resume_meta = _load_training_state(
            resume_state_dir, network, optimizer, device=device)
        if adaptive:
            adaptive.load_state_dict(_resume_meta.get("adaptive_lr_state"))
        logger.info(f"[resume] from {resume_state_dir}: continuing at epoch "
                    f"{start_epoch + 1}/{max_train_epochs} (global_step {global_step})")
        if start_epoch >= max_train_epochs:
            # Pausing ON the last epoch exits before the final LoRA is written — Resume is what
            # completes it, so this fall-through writes the final file from the restored state.
            logger.warning(f"[resume] state is at epoch {start_epoch} of {max_train_epochs} — "
                           f"nothing left to train. Writing the final LoRA from the restored "
                           f"state. To train further, raise Max Train Epochs and resume again.")

    def _meta():
        return build_metadata(
            None, ARCHITECTURE_MINIMAX, time.time(),
            title=(metadata_title if metadata_title is not None
                   else resolve_title(output_name, metadata_trigger_phrase)),
            author=metadata_author, description=metadata_description,
            license=metadata_license, tags=metadata_tags, trigger_phrase=metadata_trigger_phrase)

    def _state_extra():
        return {"adaptive_lr_state": adaptive.state_dict()} if adaptive else None

    # ---- epoch loop ----
    loss_recorder = LossRecorder()
    progress_bar = tqdm(total=steps_per_epoch * max_train_epochs, initial=global_step,
                        desc="minimax-h3")
    for epoch in range(start_epoch, max_train_epochs):
        shared_epoch.value = epoch + 1
        network.train()
        for i, batch in enumerate(loader):
            latents = batch["latents"].to(device, dtype)           # (1, 24, H, W)
            if latents.dim() == 4:
                latents = latents.unsqueeze(2)                     # -> (1, 24, 1, H, W)
            text = batch["hidden_states"].to(device, dtype)        # (1, L, 5120)
            loss, _ = compute_loss(dit, latents, text, shift=shift)
            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            loss_recorder.add(epoch=epoch, step=i, loss=loss.item())
            progress_bar.set_postfix(avr_loss=f"{loss_recorder.moving_average:.4f}", refresh=False)
            progress_bar.update(1)

        logger.info(f"epoch {epoch + 1}/{max_train_epochs} done — avr_loss {loss_recorder.moving_average:.4f}")
        if adaptive is not None:
            adaptive.epoch_boundary(epoch, loss_recorder.moving_average, network, optimizer)
        if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0 and (epoch + 1) < max_train_epochs:
            ckpt = os.path.join(output_dir, f"{output_name}-{epoch + 1:06d}.safetensors")
            _save_lora(network, ckpt, network_dim, network_alpha, dtype, _meta())
            logger.info(f"saved {ckpt}")
            if save_state:
                _save_training_state(output_dir, output_name, network, optimizer,
                                     epoch=epoch + 1, global_step=global_step,
                                     dtype=dtype, extra=_state_extra())
                prune_state_dirs(output_dir, output_name, keep_last_n_states)
        if os.path.exists(pause_flag):
            # Pause = graceful epoch-end exit with FULL state (regardless of the save-state
            # toggles), so Resume continues exactly here — matching Klein/Krea 2. The final
            # LoRA is deliberately NOT written; Resume (or the natural run end) writes it.
            _save_training_state(output_dir, output_name, network, optimizer,
                                 epoch=epoch + 1, global_step=global_step,
                                 dtype=dtype, extra=_state_extra())
            try:
                os.remove(pause_flag)
            except OSError:
                pass
            progress_bar.close()
            logger.info(f"[pause] requested — state saved at epoch {epoch + 1}. Exiting cleanly.")
            sys.exit(0)

    progress_bar.close()
    final = os.path.join(output_dir, f"{output_name}.safetensors")
    _save_lora(network, final, network_dim, network_alpha, dtype, _meta())
    logger.info(f"saved final LoRA: {final}")
    if save_state_on_train_end and max_train_epochs > start_epoch:
        _save_training_state(output_dir, output_name, network, optimizer,
                             epoch=max_train_epochs, global_step=global_step,
                             dtype=dtype, extra=_state_extra())
        prune_state_dirs(output_dir, output_name, keep_last_n_states)
    try:
        os.remove(pause_flag)
    except OSError:
        pass
    return final
