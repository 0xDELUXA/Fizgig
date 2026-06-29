"""Krea 2 LoRA training core: full-model LoRA setup + flow-matching loss.

Trains on the RAW model. The LoRA wraps all 264 Linears (no layer-targeting presets yet —
Krea2's block semantics aren't mapped, so Identity/Style/Details presets come later). The base
is frozen (optionally fp8-quantized, QLoRA-style); only the LoRA trains in bf16.
"""

import logging

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
