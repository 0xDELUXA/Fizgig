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
import os
import time
from multiprocessing import Value

import torch
import torch.nn.functional as F

from fizgig.training.metadata import ARCHITECTURE_MINIMAX

logger = logging.getLogger(__name__)

# LoRA targets the 50 transformer blocks + the 2-block text refiner ONLY — the NF4-quantized
# bf16-compute Linears. The fp32 patch/head IO layers are left alone (wrapping them clashes
# fp32-base vs bf16-adapter). This is the "Model Area to Train = full transformer" default.
DEFAULT_INCLUDE_PATTERNS = [r"blocks\..*", r"token_refiner\.blocks\..*"]


def sample_sigmas(batch: int, device, shift: float = 12.0, generator=None) -> torch.Tensor:
    """Noise levels in (0,1) with H3's resolution/video shift baked in.

    Base u ~ U(0,1), then the same sigma-shift map the model uses for its video schedule
    (sigma_shift_video=12.0): sigma = shift*u / (1 + (shift-1)*u). The shift concentrates
    training toward the high-noise end, as H3's sampler does. shift=1 recovers uniform.
    """
    u = torch.rand(batch, device=device, generator=generator)
    return (shift * u) / (1.0 + (shift - 1.0) * u)


def compute_loss(model, latent: torch.Tensor, text_embeds: torch.Tensor, *,
                 sigma: torch.Tensor = None, shift: float = 12.0, generator=None,
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
    if noise is None:
        noise = torch.randn(x0.shape, device=device, generator=generator, dtype=torch.float32)
    else:
        noise = noise.to(device=device, dtype=torch.float32)
    if sigma is None:
        sigma = sample_sigmas(1, device, shift=shift, generator=generator)
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


def _save_lora(network, path, network_dim, network_alpha, dtype, extra_metadata=None):
    metadata = {
        "ss_network_module": "fizgig.minimax (lora_unet, transformer blocks)",
        "ss_network_dim": str(network_dim),
        "ss_network_alpha": str(network_alpha),
        "ss_architecture": ARCHITECTURE_MINIMAX,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    network.save_weights(path, dtype, metadata)


def train_minimax(
    dataset_config: str,
    output_dir: str,
    output_name: str,
    dit_path: str,
    *,
    network_dim: int = 16,
    network_alpha: float = 16,
    learning_rate: float = 1e-4,
    max_train_epochs: int = 10,
    save_every_n_epochs: int = 0,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    optimizer_type: str = "adamw8bit",
    optimizer_args: str = "",
    include_patterns: list = None,
    quantize: bool = True,           # NF4 the base (QLoRA); False = bf16 base (needs ~66 GB VRAM)
    shift: float = 12.0,
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

    # ---- base (NF4-frozen) + trainable LoRA over the transformer blocks ----
    dit = load_minimax_h3_dit(dit_path, device=device, compute_dtype=dtype, quantize=quantize)
    dit.requires_grad_(False)                                   # frozen base (QLoRA-style)
    network = create_network(None, "lora_unet", 1.0, network_dim, network_alpha, None, [], dit,
                             include_patterns=include_patterns)
    network.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
    network.requires_grad_(True)
    network.to(device=device, dtype=dtype)
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

    def _meta():
        return build_metadata(
            None, ARCHITECTURE_MINIMAX, time.time(),
            title=(metadata_title if metadata_title is not None
                   else resolve_title(output_name, metadata_trigger_phrase)),
            author=metadata_author, description=metadata_description,
            license=metadata_license, tags=metadata_tags, trigger_phrase=metadata_trigger_phrase)

    # ---- epoch loop ----
    loss_recorder = LossRecorder()
    progress_bar = tqdm(total=steps_per_epoch * max_train_epochs, desc="minimax-h3")
    paused = False
    for epoch in range(max_train_epochs):
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
        if os.path.exists(pause_flag):
            logger.info("[pause] requested — saving the final LoRA and exiting cleanly.")
            paused = True
            break

    progress_bar.close()
    final = os.path.join(output_dir, f"{output_name}.safetensors")
    _save_lora(network, final, network_dim, network_alpha, dtype, _meta())
    logger.info(f"{'[pause] ' if paused else ''}saved final LoRA: {final}")
    try:
        os.remove(pause_flag)
    except OSError:
        pass
    return final
