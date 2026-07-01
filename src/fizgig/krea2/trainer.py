"""Krea 2 LoRA training: full-model LoRA + flow-matching loss over a bucketed dataloader.

Trains on the RAW model. The LoRA wraps all 264 Linears (no layer-targeting presets yet — Krea2's
block semantics aren't mapped, so Identity/Style/Details presets come later). The base is frozen
(optionally fp8, QLoRA-style); only the LoRA trains in bf16. Uses Fizgig's bucketed multi-resolution
dataloader (same framework as Klein) over the krea2 latent/TE caches.
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
from multiprocessing import Value

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fizgig.dataset.config import (
    BlueprintGenerator,
    ConfigSanitizer,
    generate_dataset_group_by_blueprint,
    load_user_config,
)
from fizgig.krea2.utils import load_krea2_dit
from fizgig.krea2.sampling import gather_valid_text, prepare
from fizgig.networks.lora import create_network
from fizgig.training.metadata import ARCHITECTURE_KREA2
from fizgig.training.train_utils import LossRecorder

logger = logging.getLogger(__name__)


def _apply_context_lora(target, path, strength, *, device, dtype):
    """Load a context LoRA and apply it FROZEN + ACTIVE on `target` (the base DiT during
    training, or the Turbo at preview time). The context and the trainable/preview LoRA each
    wrap the forward and contribute additively; gradients never flow to the context. Returns
    the network so the caller can keep a reference (and free it after previews)."""
    from safetensors.torch import load_file
    from fizgig.networks.lora import create_network_from_weights, ensure_kohya_lora_state_dict
    # Normalize foreign formats (PEFT / diffusers / ComfyUI `diffusion_model.*`, LyCORIS) to
    # kohya keys so create_network_from_weights' lora_down scan finds the modules — without
    # this a diffusers-format context LoRA yields 0 modules. Mirrors Klein's load_lora.
    sd = ensure_kohya_lora_state_dict(load_file(path))
    net = create_network_from_weights(None, float(strength), sd, None, target, for_inference=True)
    net.apply_to(text_encoders=None, unet=target, apply_text_encoder=False, apply_unet=True)
    net.load_state_dict(sd, strict=False)
    net.to(device=device, dtype=dtype).eval()
    net.requires_grad_(False)
    return net


def load_dit_for_training(
    raw_path: str,
    *,
    network_dim: int = 32,
    network_alpha: float = 32,
    fp8_scaled: bool = True,
    quant_4bit: bool = False,
    blocks_to_swap: int = 0,
    gradient_checkpointing: bool = True,
    context_lora_path: str = None,
    context_lora_strength: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load the RAW DiT (frozen base, optionally fp8) and apply a trainable full-model LoRA.
    An optional frozen Context LoRA is applied to the base first, so the new LoRA learns to
    coexist with it (the context stays active during previews too).

    quant_4bit: QLoRA-style 4-bit (NF4) frozen base — halves DiT residency (~14 GB fp8 → ~5.6 GB)
    so a full LoRA trains on a 10-12 GB card with no block swap. Mutually exclusive with block
    swap (weights live in _nf4_packed, not .weight). Loads the base bf16 on CPU and NF4-quantizes
    the block Linears onto the GPU layer-by-layer (peak VRAM never holds the whole bf16 model).
    Reuses the same target/exclude keys as the fp8 path (`blocks.` minus mod./norm/txtfusion)."""
    if quant_4bit:
        # NF4 quantizes from bf16 (cleaner than fp8->NF4 double-quant), staged on CPU, and can't
        # coexist with block swap — force both here so callers can't misconfigure it.
        fp8_scaled = False
        blocks_to_swap = 0
        loading_device = "cpu"
    else:
        loading_device = "cpu" if blocks_to_swap > 0 else device
    dit = load_krea2_dit(raw_path, device=device, dtype=dtype, fp8_scaled=fp8_scaled,
                         loading_device=loading_device)
    dit.requires_grad_(False)  # frozen base (QLoRA-style)
    if quant_4bit:
        from fizgig.krea2.utils import KREA2_FP8_OPTIMIZATION_TARGET_KEYS, KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS
        from fizgig.modules.nf4 import apply_nf4_quantization
        n_q = apply_nf4_quantization(
            dit, target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS,
            exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS, compute_device=torch.device(device))
        dit.to(device)  # move the remaining (non-quantized) bf16 modules to the GPU
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"NF4 4-bit base active: {n_q} Linears quantized; DiT resident on {device}.")
    if gradient_checkpointing:
        dit.enable_gradient_checkpointing()

    # Context LoRA: frozen + active on the base BEFORE the trainable LoRA, so the trainable
    # one wraps the context-included forward (both additive; grads only flow to the trainable).
    if context_lora_path:
        logger.info(f"context LoRA: {os.path.basename(context_lora_path)} @ {context_lora_strength} (frozen, active)")
        _apply_context_lora(dit, context_lora_path, context_lora_strength, device=device, dtype=dtype)

    network = create_network(None, "lora_unet", 1.0, network_dim, network_alpha, None, [], dit)
    network.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
    network.requires_grad_(True)
    network.to(device=device, dtype=dtype)
    return dit, network


def _get_lin_function(x1, y1, x2, y2):
    """Linear map through (x1,y1)-(x2,y2): f(x) = m*x + b. Used to schedule the flow shift `mu`
    from image-token count (musubi's get_lin_function)."""
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


# Krea 2 resolution->mu schedule (musubi `krea2_shift`): token count maps to mu, shift = exp(mu).
# Endpoints match krea2 inference defaults (minres 256, maxres 1280 at align 16):
#   x1 = (256//16)**2 = 256, x2 = (1280//16)**2 = 6400, y1 = 0.5, y2 = 1.15.
_KREA2_MU = _get_lin_function(256, 0.5, 6400, 1.15)


def sample_krea2_timesteps(bsize: int, num_img_tokens: int, device, sigmoid_scale: float = 1.0) -> torch.Tensor:
    """Krea 2 'krea2_shift' timestep sampling — a faithful port of the musubi krea2_train recipe.

    The base t is **logit-normal** (sigmoid of a standard normal), so timesteps concentrate near the
    middle instead of being uniform. Uniform sampling (the old code) dumps far too much mass on the
    high-noise end, where the flow-matching velocity is intrinsically hard to predict — that inflates
    the loss AND skews the training signal away from the validated reference recipe. The shift is
    resolution-dependent (shift = exp(mu), mu from the image-token count), not a fixed 2.5.

        t_base = sigmoid(randn * sigmoid_scale)
        t      = (t_base * shift) / (1 + (shift - 1) * t_base)
    """
    mu = _KREA2_MU(num_img_tokens)
    shift = math.exp(mu)
    t = (torch.randn(bsize, device=device) * sigmoid_scale).sigmoid()
    return (t * shift) / (1.0 + (shift - 1.0) * t)


def compute_loss(dit, latent, hidden_states, attention_mask, *, shift=2.5, dtype=torch.bfloat16):
    """Flow-matching training loss for Krea 2.

    latent:        (B, 16, h, w)         — cached Qwen-Image VAE latent
    hidden_states: (B, seq, layers, dim) — cached Qwen3-VL multi-layer stack
    attention_mask:(B, seq) bool         — cached validity mask

    `shift` is kept for signature compatibility but no longer used: krea2_shift derives the flow
    shift from the image resolution (see sample_krea2_timesteps), matching the musubi reference.
    """
    device = next(p for p in dit.parameters()).device
    B = latent.shape[0]
    latent = latent.to(device=device, dtype=dtype)
    patch = dit.config.patch

    noise = torch.randn_like(latent)
    # krea2_shift: logit-normal base + resolution-dependent shift, over the image-token count
    # (latent grid // patch). Replaces the old uniform-u sampler that over-weighted high-noise t
    # and inflated the loss.
    num_img_tokens = (latent.shape[-2] // patch) * (latent.shape[-1] // patch)
    t = sample_krea2_timesteps(B, num_img_tokens, device)
    t_ = t.view(B, 1, 1, 1).to(dtype)
    noised = (1.0 - t_) * latent + t_ * noise
    target = noise - latent  # flow-matching velocity

    txt, txtmask = gather_valid_text(hidden_states.to(device=device, dtype=dtype), attention_mask.to(device))
    img_tokens, pos, mask = prepare(noised, txt.shape[1], patch, txtmask)
    target_tokens, _, _ = prepare(target, txt.shape[1], patch, txtmask)

    with torch.autocast(device_type=torch.device(device).type, dtype=dtype):
        pred = dit(img=img_tokens, context=txt, t=t.to(dtype), pos=pos, mask=mask)
    return F.mse_loss(pred.float(), target_tokens.float())


class _Krea2Collator:
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
                         network_dim, network_alpha, dtype, extra=None):
    """Save a resumable training-state dir matching Klein's naming: <name>-<NNNNNN>-state/.

    NNNNNN is the number of COMPLETED epochs (= the next 0-indexed epoch to run). The dir
    holds the LoRA weights, the optimizer state, RNG states, and a small JSON. The GUI's
    _detect_latest_state_dir finds the highest-numbered one and passes it to --resume."""
    state_dir = os.path.join(output_dir, f"{output_name}-{epoch:06d}-state")
    os.makedirs(state_dir, exist_ok=True)
    _save_lora(network, os.path.join(state_dir, "lora.safetensors"), network_dim, network_alpha, dtype)
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
    from safetensors.torch import load_file
    network.load_state_dict(load_file(os.path.join(state_dir, "lora.safetensors")), strict=False)
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


class AdaptiveLR:
    """Bi-directional plateau LR tracker — a faithful port of Klein's adaptive_lr logic.

    Each epoch boundary: probe UP ×1.25 on steady loss descent (patience 2); reduce DOWN ×0.5
    on loss plateau (patience ramp) or a stability signal. On a stability event it blends the
    LoRA weights 70/30 toward the previous epoch's snapshot and restores the optimizer state
    (kills bad Adam momentum). Klein's stability signals are grad-clip ratio + weight-norm
    growth; krea2 has no grad clipping, so weight-norm growth (>30%) is the stability signal.

    State (streaks/best_loss/prev_weight_norm) is JSON round-trippable for pause/resume; the
    CPU rollback snapshot is in-memory only (too big to persist) — so the first post-resume
    epoch can't roll back, exactly as in Klein. Call epoch_boundary() at each epoch end."""

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


def _save_lora(network, path, network_dim, network_alpha, dtype, extra_metadata=None):
    metadata = {
        "ss_network_module": "fizgig.krea2 (lora_unet, all-Linear)",
        "ss_network_dim": str(network_dim),
        "ss_network_alpha": str(network_alpha),
        "ss_architecture": ARCHITECTURE_KREA2,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    network.save_weights(path, dtype, metadata)


# --- in-training previews (sample the fp8 Turbo with the live LoRA) -----------
def encode_sample_prompts(te_path, prompts, *, ref_image=None, vision_megapixels=1.0, device="cuda"):
    """Pre-encode the sample prompts once (Qwen3-VL), freeing the encoder afterwards.
    Returns a list of (txt, txtmask) on CPU, fed straight to sampling.sample at preview time.

    `ref_image` (a PIL image or path) routes a reference through Qwen3-VL's vision path so the
    samples become visually aware of it ('prompt from a picture' — Krea 2's reference mechanism)."""
    from fizgig.krea2.utils import load_krea2_text_encoder
    from fizgig.krea2 import sampling

    pil = None
    if ref_image:
        from PIL import Image
        pil = ref_image if hasattr(ref_image, "convert") else Image.open(ref_image)

    enc = load_krea2_text_encoder(te_path, dtype=torch.bfloat16, device=device)
    out = []
    for p in prompts:
        images = [[pil]] if pil is not None else None
        txt, txtmask, _, _ = sampling.encode_prompts(enc, [p], cfg=False,
                                                     images=images, vision_megapixels=vision_megapixels)
        out.append((txt.cpu(), txtmask.cpu()))
    del enc
    torch.cuda.empty_cache()
    return out


def _read_sample_override(output_dir):
    """Live sample override written by the GUI to <output_dir>/.sample_override.json.

    Returns {prompt, seed, width, height, ref_image} while active, else None. ref_image (if set)
    is routed through Qwen3-VL's vision path (Krea 2's reference mechanism)."""
    path = os.path.join(output_dir, ".sample_override.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        prompt = str(d.get("prompt", "")).strip()
        ref = str(d.get("ref_image", "")).strip()
        # Active on a prompt OR a reference — a reference with an empty prompt is a valid
        # 'generate from this picture' override (the Qwen3-VL vision path handles the rest).
        if prompt or ref:
            return {"prompt": prompt,
                    "seed": int(d.get("seed", 1234)),
                    "width": int(d.get("width", 1024)),
                    "height": int(d.get("height", 1024)),
                    "ref_image": ref}
    except Exception:
        pass
    return None


def sample_previews(turbo_path, ae, encoded_prompts, lora_sd, out_dir, epoch, *,
                    output_name="krea2", steps=8, cfg_scale=1.0, width=512, height=512,
                    seed=42, context_lora_path=None, context_lora_strength=1.0,
                    blocks_to_swap=0, int8=False, device="cuda"):
    """Load the (clean) pre-quant fp8 Turbo, apply the current LoRA LIVE (no merge -> no grid),
    and render each pre-encoded prompt. Turbo is freed afterwards.

    `blocks_to_swap` > 0 puts the Turbo on forward-only block swap so previews fit smaller cards
    (mirrors Klein's Distilled sample-model auto-swap). Order mirrors load_dit_for_training: load
    the base on CPU, apply the LoRA(s), then enable swap + place the resident blocks.

    Filenames follow the Fizgig samples-gallery pattern
    `{name}_e{epoch:06d}_{idx:02d}_{timestamp:14d}_{seed}.png` so the live preview gallery
    (which parses that exact format) picks them up — same as the Klein training path."""
    import datetime
    from fizgig.krea2.utils import load_krea2_dit
    from fizgig.networks.lora import create_network_from_weights
    from fizgig.krea2 import sampling

    _ld = "cpu" if blocks_to_swap > 0 else device
    turbo = load_krea2_dit(turbo_path, device=device, dtype=torch.bfloat16,
                           loading_device=_ld)  # prequant fp8 auto-detected
    if int8:
        # INT8 (W8A8) fast preview matmul — quantize the block Linears BEFORE the LoRA wraps them
        # (so the LoRA wraps the int8 forward) and before block swap (so the offloader stages int8).
        # Quantize on the load device so a swapped (CPU-loaded) model doesn't need the whole int8
        # model resident on GPU.
        from fizgig.modules.int8 import apply_int8_quantization
        from fizgig.krea2.utils import (KREA2_FP8_OPTIMIZATION_TARGET_KEYS,
                                        KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS)
        apply_int8_quantization(turbo, target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS,
                                exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS,
                                compute_device=torch.device(_ld))
    # Context LoRA (frozen) goes on FIRST so previews match deployment: the trained LoRA runs
    # on top of the same context at the same strength it was trained with.
    ctx_net = None
    if context_lora_path:
        ctx_net = _apply_context_lora(turbo, context_lora_path, context_lora_strength,
                                      device=device, dtype=torch.bfloat16)
    net = create_network_from_weights(None, 1.0, lora_sd, None, turbo, for_inference=True)
    net.apply_to(text_encoders=None, unet=turbo, apply_text_encoder=False, apply_unet=True)
    # create_network_from_weights only builds the module STRUCTURE (sizes from dims/alphas);
    # the trained values must be loaded in, or the LoRA stays at its zero init (lora_up=0) and
    # contributes nothing — which made every epoch's preview identical. Mirrors the Klein path
    # (inference.py: apply_to -> load_state_dict(strict=False)).
    net.load_state_dict(lora_sd, strict=False)
    net.to(device=device, dtype=torch.bfloat16).eval()
    if blocks_to_swap > 0:
        from fizgig.krea2.offloading import BlockSwapConfig
        turbo.enable_block_swap(blocks_to_swap, BlockSwapConfig(torch.device(device), supports_backward=False))
        turbo.move_to_device_except_swap_blocks(torch.device(device))
        turbo.switch_block_swap_for_inference()
    turbo.eval()
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 14-digit timestamp
    paths = []
    for i, (txt, txtmask) in enumerate(encoded_prompts):
        with torch.no_grad():
            imgs = sampling.sample(turbo, ae, txt, txtmask, untxt=None, untxtmask=None,
                                   device=device, dtype=torch.bfloat16, width=width, height=height,
                                   steps=steps, cfg_scale=cfg_scale, mu=1.15, seed=seed + i)
        p = os.path.join(out_dir, f"{output_name}_e{epoch:06d}_{i:02d}_{ts}_{seed + i}.png")
        imgs[0].save(p)
        paths.append(p)
    del turbo, net, ctx_net
    torch.cuda.empty_cache()
    return paths


def train_krea2(
    raw_path: str,
    dataset_config: str,
    output_dir: str,
    output_name: str,
    *,
    network_dim: int = 32,
    network_alpha: float = 32,
    learning_rate: float = 1e-4,
    max_train_epochs: int = 10,
    save_every_n_epochs: int = 0,
    fp8_scaled: bool = True,
    quant_4bit: bool = False,
    blocks_to_swap: int = 0,
    shift: float = 2.5,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    # in-training previews (sample the fp8 Turbo with the live LoRA)
    sample_prompts: list = None,
    turbo_path: str = None,
    vae_path: str = None,
    te_path: str = None,
    sample_every_n_epochs: int = 0,
    sample_width: int = 512,
    sample_height: int = 512,
    sample_steps: int = 8,
    sample_seed: int = 42,
    sample_ref_image: str = None,
    preview_blocks_to_swap: int = 0,
    preview_int8: bool = False,
    resume_state_dir: str = None,
    context_lora_path: str = None,
    context_lora_strength: float = 1.0,
    adaptive_lr: bool = False,
    adaptive_lr_min: float = 1e-5,
    adaptive_lr_max: float = 4e-4,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Native Krea 2 LoRA training: bucketed multi-resolution dataloader over the krea2 caches ->
    flow-matching loss -> AdamW -> save a ComfyUI-compatible LoRA. In-training Turbo previews +
    GUI wiring are layered on elsewhere."""
    torch.manual_seed(seed)

    shared_epoch = Value("i", 0)
    user_config = load_user_config(dataset_config)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        user_config, argparse.Namespace(), architecture=ARCHITECTURE_KREA2)
    group = generate_dataset_group_by_blueprint(
        blueprint.dataset_group, training=True, num_timestep_buckets=None, shared_epoch=shared_epoch)
    if group.num_train_items == 0:
        raise RuntimeError("No training items — run the krea2 cache scripts first.")
    logger.info(f"Krea 2 training: {group.num_train_items} items, {max_train_epochs} epochs")

    # Preview setup: pre-encode prompts (frees the 8GB encoder) + load the VAE BEFORE the RAW DiT,
    # so the encoder never coexists with the resident base.
    do_previews = bool(sample_every_n_epochs and sample_prompts and turbo_path and vae_path and te_path)
    encoded_prompts = sample_ae = sample_dir = None
    if do_previews:
        from fizgig.krea2.vae_loader import load_vae
        logger.info(f"pre-encoding {len(sample_prompts)} sample prompt(s)"
                    f"{' with reference image' if sample_ref_image else ''}...")
        encoded_prompts = encode_sample_prompts(te_path, sample_prompts, ref_image=sample_ref_image, device=device)
        sample_ae = load_vae(vae_path, input_channels=3, device="cpu", disable_mmap=True)
        sample_dir = os.path.join(output_dir, "sample")

    if quant_4bit and blocks_to_swap > 0:
        logger.info("[nf4] 4-bit base is incompatible with block swap (weights live in _nf4_packed) "
                    "— forcing blocks_to_swap=0.")
        blocks_to_swap = 0
    dit, network = load_dit_for_training(
        raw_path, network_dim=network_dim, network_alpha=network_alpha,
        fp8_scaled=fp8_scaled, quant_4bit=quant_4bit, blocks_to_swap=blocks_to_swap,
        context_lora_path=context_lora_path, context_lora_strength=context_lora_strength,
        device=device, dtype=dtype)
    if blocks_to_swap > 0 and not quant_4bit:
        from fizgig.krea2.offloading import BlockSwapConfig
        dit.enable_block_swap(blocks_to_swap, BlockSwapConfig(torch.device(device), supports_backward=True))
        dit.move_to_device_except_swap_blocks(torch.device(device))
        dit.switch_block_swap_for_training()
    dit.train()
    network.train()
    network.requires_grad_(True)

    params = list(network.get_trainable_params())
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(params, lr=learning_rate)
        logger.info("optimizer: AdamW8bit")
    except Exception:
        optimizer = torch.optim.AdamW(params, lr=learning_rate)
        logger.info("optimizer: AdamW (bitsandbytes unavailable)")

    collator = _Krea2Collator(shared_epoch, group)
    loader = DataLoader(group, batch_size=1, shuffle=True, collate_fn=collator, num_workers=0)

    os.makedirs(output_dir, exist_ok=True)
    adaptive = AdaptiveLR(adaptive_lr_min, adaptive_lr_max) if adaptive_lr else None
    if adaptive:
        logger.info(f"[adaptive_lr] ENABLED — start_lr={learning_rate:.3e} "
                    f"min_lr={adaptive_lr_min:.3e} max_lr={adaptive_lr_max:.3e}")

    global_step = 0
    start_epoch = 0
    # Resume: restore LoRA + optimizer + RNG + (start_epoch, global_step) from a saved state dir.
    if resume_state_dir and os.path.isdir(resume_state_dir):
        start_epoch, global_step, _resume_meta = _load_training_state(resume_state_dir, network, optimizer, device=device)
        if adaptive:
            adaptive.load_state_dict(_resume_meta.get("adaptive_lr_state"))
            logger.info(f"[resume] adaptive_lr state restored: best_loss={adaptive.best_loss} "
                        f"streaks g/b/s={adaptive.good_streak}/{adaptive.bad_streak}/{adaptive.stability_streak} "
                        f"stability_triggered={adaptive.stability_triggered}")
        logger.info(f"[resume] from {resume_state_dir}: continuing at epoch {start_epoch + 1}/{max_train_epochs} "
                    f"(global_step {global_step})")
    try:
        steps_per_epoch = len(loader)
    except TypeError:
        steps_per_epoch = group.num_train_items
    pause_flag = os.path.join(output_dir, ".pause_requested")
    # Progress + loss display exactly as Klein: one continuous tqdm bar over all steps with
    # a smoothed avr_loss in the postfix (the raw per-step loss is very noisy — batch size 1
    # plus a random flow-matching timestep each step — so the moving average is the signal).
    loss_recorder = LossRecorder()
    progress_bar = tqdm(total=steps_per_epoch * max_train_epochs, initial=global_step,
                        desc="steps", smoothing=0)
    for epoch in range(start_epoch, max_train_epochs):
        shared_epoch.value = epoch + 1
        for i, batch in enumerate(loader):
            loss = compute_loss(dit, batch["latents"], batch["hidden_states"], batch["attention_mask"],
                                shift=shift, dtype=dtype)
            loss.backward()
            # Gradient clipping to match the musubi reference (max_grad_norm default 1.0). 0 disables.
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            loss_recorder.add(epoch=epoch, step=i, loss=loss.item())
            # refresh=False so only update(1) draws the bar — otherwise set_postfix AND update each
            # force a refresh, which a captured (non-tty) stderr logs as two lines per step (the
            # "187, 187, 188, 188" doubling). Training itself is one step per iteration.
            progress_bar.set_postfix(avr_loss=f"{loss_recorder.moving_average:.4f}", refresh=False)
            progress_bar.update(1)
        logger.info(f"epoch {epoch + 1}/{max_train_epochs}  avr_loss={loss_recorder.moving_average:.4f}  step={global_step}")

        # Adaptive LR: epoch-boundary plateau tracker (before save/preview so they reflect the
        # post-adjustment state). Uses the smoothed avr_loss as the signal, like Klein.
        if adaptive:
            adaptive.epoch_boundary(epoch, loss_recorder.moving_average, network, optimizer)

        if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0 and (epoch + 1) < max_train_epochs:
            _save_lora(network, os.path.join(output_dir, f"{output_name}-{epoch + 1:06d}.safetensors"),
                       network_dim, network_alpha, dtype)

        if do_previews and (epoch + 1) % sample_every_n_epochs == 0:
            from safetensors.torch import load_file
            tmp = os.path.join(output_dir, "_sample_lora.safetensors")
            _save_lora(network, tmp, network_dim, network_alpha, dtype)
            logger.info(f"rendering previews (epoch {epoch + 1}) on the fp8 Turbo...")
            # The preview loads the fp8 Turbo (~13 GB) on top of the resident training DiT
            # (~14 GB fp8) + the VAE — two full models won't fit (OOMs ~30 GB on a 32 GB card).
            # Park the training DiT on CPU for the preview, then restore it (and its block-swap
            # placement) before the next epoch. Costs one CPU<->GPU round-trip per preview.
            dit.to("cpu")
            if getattr(dit, "_nf4_quantized", False):
                # NF4's packed weights + quant state are plain attributes that .to("cpu") ignores
                # (~6 GB would stay on the GPU), so move them explicitly to free the VRAM the
                # preview needs — restored in the finally below.
                from fizgig.modules.nf4 import move_nf4_to_device
                move_nf4_to_device(dit, "cpu")
            gc.collect()
            torch.cuda.empty_cache()
            try:
                # Live sample override (GUI status-bar panel) — model-agnostic prompt/seed/res
                # for the next preview. Encoded here (after the training DiT is on CPU) so the
                # text encoder has room. No override -> the configured pre-encoded prompts.
                ov = _read_sample_override(output_dir)
                if ov:
                    logger.info(f"[sample override] active — '{ov['prompt'][:60]}' "
                                f"seed={ov['seed']} {ov['width']}x{ov['height']}"
                                f"{' +ref' if ov.get('ref_image') else ''}")
                    prev_enc = encode_sample_prompts(te_path, [ov["prompt"]],
                                                     ref_image=ov.get("ref_image") or None, device=device)
                    prev_w, prev_h, prev_seed = ov["width"], ov["height"], ov["seed"]
                else:
                    prev_enc, prev_w, prev_h, prev_seed = encoded_prompts, sample_width, sample_height, sample_seed
                sample_previews(turbo_path, sample_ae, prev_enc, load_file(tmp), sample_dir, epoch + 1,
                                output_name=output_name, steps=sample_steps, width=prev_w,
                                height=prev_h, seed=prev_seed,
                                context_lora_path=context_lora_path, context_lora_strength=context_lora_strength,
                                blocks_to_swap=preview_blocks_to_swap, int8=preview_int8, device=device)
            except Exception as _prev_err:
                # A preview failure — almost always CUDA OOM (the ~13 GB Turbo + the Qwen3-VL
                # encoder won't fit alongside the parked training DiT on a small card) — must NEVER
                # kill the run. Training and LoRA saving are independent of previews, so we log,
                # disable previews for the rest of this run (so we don't re-OOM every sample epoch),
                # and carry on. The training DiT is restored in the finally below.
                _oom = "out of memory" in str(_prev_err).lower()
                logger.warning(
                    f"[preview] epoch {epoch + 1} preview failed "
                    f"({'CUDA OOM — this card is too small for the Turbo preview' if _oom else type(_prev_err).__name__}); "
                    f"disabling previews for the rest of the run. Training continues and LoRAs still save normally."
                )
                do_previews = False
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                if blocks_to_swap > 0:
                    # Re-establish the training placement (non-swap blocks -> GPU, swap blocks -> CPU).
                    dit.move_to_device_except_swap_blocks(torch.device(device))
                    dit.switch_block_swap_for_training()
                else:
                    dit.to(device)
                if getattr(dit, "_nf4_quantized", False):
                    # Restore the 4-bit packed weights + quant state to the GPU (they were parked
                    # on CPU above; .to(device) doesn't touch them). NF4 forces blocks_to_swap=0.
                    from fizgig.modules.nf4 import move_nf4_to_device
                    move_nf4_to_device(dit, device)
            dit.train()
            network.train()

        # Graceful pause (GUI wrote <output_dir>/.pause_requested): save a full resumable
        # state at this epoch boundary and exit cleanly so the GPU frees. The GUI detects the
        # clean exit, records the paused state, and offers Resume. Same contract as Klein.
        if os.path.exists(pause_flag):
            logger.info(f"[pause] requested — saving state at epoch {epoch + 1} and exiting cleanly")
            _save_training_state(output_dir, output_name, network, optimizer,
                                 epoch=epoch + 1, global_step=global_step,
                                 network_dim=network_dim, network_alpha=network_alpha, dtype=dtype,
                                 extra={"adaptive_lr_state": adaptive.state_dict()} if adaptive else None)
            try:
                os.remove(pause_flag)
            except Exception:
                pass
            progress_bar.close()
            logger.info("[pause] state saved — exiting (exit 0).")
            sys.exit(0)

    progress_bar.close()
    out = os.path.join(output_dir, f"{output_name}.safetensors")
    # Record the context LoRA in metadata so users know to pair it at the same strength at
    # inference (the trained LoRA is context-dependent — same contract as Klein).
    extra = None
    if context_lora_path:
        extra = {"ss_context_lora": os.path.basename(context_lora_path),
                 "ss_context_lora_strength": str(context_lora_strength)}
    _save_lora(network, out, network_dim, network_alpha, dtype, extra_metadata=extra)
    logger.info(f"saved final LoRA -> {out}")
    return out
