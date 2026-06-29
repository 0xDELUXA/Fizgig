"""Krea 2 LoRA training core: full-model LoRA setup + flow-matching loss.

Trains on the RAW model. The LoRA wraps all 264 Linears (no layer-targeting presets yet —
Krea2's block semantics aren't mapped, so Identity/Style/Details presets come later). The base
is frozen (optionally fp8-quantized, QLoRA-style); only the LoRA trains in bf16.
"""

import logging
import random

import torch
import torch.nn.functional as F

from fizgig.krea2.utils import load_krea2_dit
from fizgig.krea2.sampling import gather_valid_text, prepare
from fizgig.networks.lora import create_network

logger = logging.getLogger(__name__)


def load_dit_for_training(
    raw_path: str,
    *,
    network_dim: int = 32,
    network_alpha: float = 32,
    fp8_scaled: bool = True,
    blocks_to_swap: int = 0,
    gradient_checkpointing: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load the RAW DiT (frozen base, optionally fp8) and apply a trainable full-model LoRA.

    For block swap, pass blocks_to_swap>0; the base loads to CPU and resident blocks move to
    `device`, with swap blocks streamed during forward (set up by the caller via the model's
    enable_block_swap / move_to_device_except_swap_blocks like inference)."""
    loading_device = "cpu" if blocks_to_swap > 0 else device
    dit = load_krea2_dit(raw_path, device=device, dtype=dtype, fp8_scaled=fp8_scaled,
                         loading_device=loading_device)
    dit.requires_grad_(False)  # frozen base (QLoRA-style)
    if gradient_checkpointing:
        dit.enable_gradient_checkpointing()

    network = create_network(None, "lora_unet", 1.0, network_dim, network_alpha, None, [], dit)
    network.apply_to(text_encoders=None, unet=dit, apply_text_encoder=False, apply_unet=True)
    network.requires_grad_(True)
    network.to(device=device, dtype=dtype)
    return dit, network


def sample_shifted_timesteps(bsize: int, shift: float, device) -> torch.Tensor:
    """Flow-matching 'shift' timestep sampling: u~U(0,1), t = shift*u / (1 + (shift-1)*u)."""
    u = torch.rand(bsize, device=device)
    return shift * u / (1.0 + (shift - 1.0) * u)


def compute_loss(
    dit,
    latent: torch.Tensor,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    shift: float = 2.5,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Flow-matching training loss for Krea 2.

    latent:        (B, 16, h, w)        — cached Qwen-Image VAE latent
    hidden_states: (B, seq, layers, dim) — cached Qwen3-VL multi-layer stack
    attention_mask:(B, seq) bool         — cached validity mask
    """
    device = next(p for p in dit.parameters()).device
    B = latent.shape[0]
    latent = latent.to(device=device, dtype=dtype)

    noise = torch.randn_like(latent)
    t = sample_shifted_timesteps(B, shift, device)
    t_ = t.view(B, 1, 1, 1).to(dtype)
    noised = (1.0 - t_) * latent + t_ * noise
    target = noise - latent  # flow-matching velocity

    txt, txtmask = gather_valid_text(hidden_states.to(device=device, dtype=dtype), attention_mask.to(device))
    patch = dit.config.patch
    img_tokens, pos, mask = prepare(noised, txt.shape[1], patch, txtmask)
    target_tokens, _, _ = prepare(target, txt.shape[1], patch, txtmask)

    with torch.autocast(device_type=torch.device(device).type, dtype=dtype):
        pred = dit(img=img_tokens, context=txt, t=t.to(dtype), pos=pos, mask=mask)
    return F.mse_loss(pred.float(), target_tokens.float())


# --- minimal training loop (reads krea2 caches directly) ---------------------
def find_cache_pairs(cache_dir):
    """Pair (latent_cache, te_cache) by item key from a krea2 cache directory."""
    import glob
    import os
    import re

    latents, tes = {}, {}
    for f in glob.glob(os.path.join(cache_dir, "*.safetensors")):
        b = os.path.basename(f)
        if b.endswith("_krea2_te.safetensors"):
            tes[b[: -len("_krea2_te.safetensors")]] = f
        elif b.endswith("_krea2.safetensors"):
            m = re.match(r"^(.*)_\d+x\d+_krea2\.safetensors$", b)
            if m:
                latents[m.group(1)] = f
    return [(latents[k], tes[k]) for k in sorted(latents) if k in tes]


def _load_cached_item(latent_path, te_path):
    from safetensors.torch import load_file

    lf = load_file(latent_path)
    latent = next(v for k, v in lf.items() if k.startswith("latent_"))  # (16, h, w)
    tf = load_file(te_path)
    return latent, tf["hidden_states"], tf["attention_mask"].to(torch.bool)


def train_krea2(
    raw_path: str,
    cache_dir: str,
    output_dir: str,
    output_name: str,
    *,
    network_dim: int = 32,
    network_alpha: float = 32,
    learning_rate: float = 1e-4,
    max_train_epochs: int = 10,
    num_repeats: int = 1,
    save_every_n_epochs: int = 0,
    fp8_scaled: bool = True,
    blocks_to_swap: int = 0,
    shift: float = 2.5,
    seed: int = 42,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Minimal native Krea 2 LoRA training loop: cache pairs -> flow-matching loss -> AdamW -> save.

    Reads the krea2 latent/TE caches directly (batch size 1). Bucketing / multi-resolution
    batching and in-training Turbo previews are deliberately left for the GUI-driven trainer;
    this is the verified core loop."""
    import os

    torch.manual_seed(seed)
    pairs = find_cache_pairs(cache_dir)
    if not pairs:
        raise RuntimeError(f"No krea2 cache pairs found in {cache_dir} — run the cache scripts first.")
    logger.info(f"Krea 2 training: {len(pairs)} cached items x {num_repeats} repeats, {max_train_epochs} epochs")

    dit, network = load_dit_for_training(
        raw_path, network_dim=network_dim, network_alpha=network_alpha,
        fp8_scaled=fp8_scaled, blocks_to_swap=blocks_to_swap, device=device, dtype=dtype,
    )
    if blocks_to_swap > 0:
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

    os.makedirs(output_dir, exist_ok=True)
    items = pairs * num_repeats
    global_step = 0
    for epoch in range(max_train_epochs):
        random.shuffle(items)
        epoch_loss = 0.0
        for latent_path, te_path in items:
            latent, hidden, mask = _load_cached_item(latent_path, te_path)
            loss = compute_loss(dit, latent.unsqueeze(0), hidden.unsqueeze(0), mask.unsqueeze(0),
                                shift=shift, dtype=dtype)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item()
            global_step += 1
        epoch_loss /= len(items)
        logger.info(f"epoch {epoch + 1}/{max_train_epochs}  loss={epoch_loss:.4f}  step={global_step}")

        if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0 and (epoch + 1) < max_train_epochs:
            _save_lora(network, os.path.join(output_dir, f"{output_name}-{epoch + 1:06d}.safetensors"),
                       network_dim, network_alpha, dtype)

    out = os.path.join(output_dir, f"{output_name}.safetensors")
    _save_lora(network, out, network_dim, network_alpha, dtype)
    logger.info(f"saved final LoRA -> {out}")
    return out


def _save_lora(network, path, network_dim, network_alpha, dtype):
    from fizgig.training.metadata import ARCHITECTURE_KREA2

    metadata = {
        "ss_network_module": "fizgig.krea2 (lora_unet, all-Linear)",
        "ss_network_dim": str(network_dim),
        "ss_network_alpha": str(network_alpha),
        "ss_architecture": ARCHITECTURE_KREA2,
    }
    network.save_weights(path, dtype, metadata)
