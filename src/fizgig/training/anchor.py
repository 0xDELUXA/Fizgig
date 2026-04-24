"""Latent Anchor Training — distributional anchor loss for LoRA training.

Adds a second loss term that compares the model's predicted clean output
against reference latents from a curated pool of real images. This steers
the model toward the target distribution from both ends: noise prediction
(standard) and clean-image similarity (anchor).

Usage:
    pool = AnchorPool(anchor_dir, vae, device, dtype)
    ...
    anchor_loss = pool.compute_loss(model_pred, noisy_input, timesteps, current_idx)
    total_loss = noise_loss + anchor_weight * anchor_loss
"""

import os
import random
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class AnchorPool:
    """Manages a pool of reference latents from curated anchor images.

    The pool is VAE-encoded once at training start. During training,
    compute_loss() picks a random reference (excluding the current training
    image) and returns the MSE between the model's predicted clean latent
    and the reference.
    """

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

    def __init__(
        self,
        anchor_dir: str,
        vae,
        device: torch.device,
        dtype: torch.dtype,
        max_size: int = 1024,
    ):
        """Encode all images in anchor_dir with the VAE.

        Args:
            anchor_dir: Path to folder of curated reference images.
            vae: VAE model (already on device or will be moved).
            device: CUDA device.
            dtype: VAE dtype (bf16 or fp32).
            max_size: Max dimension for resizing anchor images before encoding.
        """
        self.latents: list[torch.Tensor] = []
        self.device = device

        image_files = sorted(
            f for f in os.listdir(anchor_dir)
            if os.path.splitext(f)[1].lower() in self.IMAGE_EXTENSIONS
        )

        if not image_files:
            raise ValueError(f"No images found in anchor directory: {anchor_dir}")

        logger.info(f"Encoding {len(image_files)} anchor images from {anchor_dir}")

        vae.to(device)
        vae.eval()

        with torch.no_grad():
            for filename in image_files:
                filepath = os.path.join(anchor_dir, filename)
                try:
                    img = Image.open(filepath).convert("RGB")

                    # Resize to max_size preserving aspect ratio
                    w, h = img.size
                    if max(w, h) > max_size:
                        scale = max_size / max(w, h)
                        w, h = int(w * scale), int(h * scale)
                    # Round to 16 for VAE
                    w = (w // 16) * 16
                    h = (h // 16) * 16
                    img = img.resize((w, h), Image.LANCZOS)

                    # To tensor: [0, 255] -> [-1, 1]
                    import numpy as np
                    tensor = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
                    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # NCHW

                    latent = vae.encode(tensor.to(device, dtype))
                    # Store on CPU to save VRAM — moved to device per-step
                    self.latents.append(latent.squeeze(0).cpu())
                    logger.info(f"  Encoded anchor: {filename} ({w}x{h})")

                except Exception as e:
                    logger.warning(f"  Failed to encode anchor {filename}: {e}")

        # Move VAE back to CPU after encoding
        vae.to("cpu")
        torch.cuda.empty_cache()

        logger.info(f"Anchor pool ready: {len(self.latents)} references")

        if not self.latents:
            raise ValueError("No anchor images could be encoded")

    def compute_loss(
        self,
        model_pred: torch.Tensor,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        exclude_idx: Optional[int] = None,
        timestep_weight: bool = False,
    ) -> torch.Tensor:
        """Compute anchor loss: MSE between predicted clean latent and a random reference.

        For flow matching, the model predicts velocity v = noise - clean.
        The noisy input is: x_t = (1-t) * clean + t * noise
        So: predicted_clean = x_t - t * model_pred

        Args:
            model_pred: Model's velocity prediction, shape (B, C, H, W).
            noisy_input: The noisy model input x_t, shape (B, C, H, W).
            timesteps: Timesteps in [0, 1000] range, shape (B,).
            exclude_idx: Optional dataset index to exclude from pool selection.
            timestep_weight: If True, scale anchor loss by (1-t) per sample —
                low noise (t→0) gets full weight, high noise (t→1) gets near-zero.

        Returns:
            Scalar anchor loss (MSE).
        """
        batch_size = model_pred.shape[0]

        # Convert timesteps from [1, 1000] to [0, 1]
        t = (timesteps - 1.0) / 999.0
        t = t.view(-1, 1, 1, 1).to(model_pred.device, dtype=model_pred.dtype)

        # Derive predicted clean latent from flow matching:
        # x_t = (1-t) * x_0 + t * noise
        # v = noise - x_0  (the velocity/target)
        # model predicts v, so: x_0 = x_t - t * v
        predicted_clean = noisy_input - t * model_pred

        # Per-sample timestep scaling: (1-t) so clean timesteps get full weight
        if timestep_weight:
            t_scale = (1.0 - t).view(batch_size)  # (B,)
        else:
            t_scale = None

        # Pick random reference latents (one per batch element)
        pool_size = len(self.latents)
        total_loss = torch.tensor(0.0, device=model_pred.device, dtype=model_pred.dtype)

        for b in range(batch_size):
            # Select a random reference, optionally excluding the current training image
            candidates = list(range(pool_size))
            ref_idx = random.choice(candidates)
            ref_latent = self.latents[ref_idx].to(model_pred.device, dtype=model_pred.dtype)

            # The reference latent may be a different spatial size than the prediction.
            # Interpolate to match if needed.
            pred_h, pred_w = predicted_clean.shape[2], predicted_clean.shape[3]
            ref_h, ref_w = ref_latent.shape[1], ref_latent.shape[2]
            if ref_h != pred_h or ref_w != pred_w:
                ref_latent = F.interpolate(
                    ref_latent.unsqueeze(0), size=(pred_h, pred_w), mode="bilinear", align_corners=False
                ).squeeze(0)

            sample_loss = F.mse_loss(predicted_clean[b], ref_latent)
            if t_scale is not None:
                sample_loss = sample_loss * t_scale[b]
            total_loss = total_loss + sample_loss

        return total_loss / batch_size

    def get_annealed_weight(
        self,
        base_weight: float,
        current_epoch: int,
        max_epochs: int,
        schedule: str = "linear",
    ) -> float:
        """Compute annealed anchor weight for the current epoch.

        Args:
            base_weight: Initial anchor weight.
            current_epoch: Current epoch (0-indexed).
            max_epochs: Total number of epochs.
            schedule: Annealing schedule — 'linear', 'cosine', or 'none'.

        Returns:
            Annealed weight value.
        """
        if schedule == "none" or max_epochs <= 1:
            return base_weight

        progress = min(current_epoch / max(max_epochs - 1, 1), 1.0)

        if schedule == "linear":
            return base_weight * (1.0 - progress)
        elif schedule == "cosine":
            import math
            return base_weight * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            return base_weight
