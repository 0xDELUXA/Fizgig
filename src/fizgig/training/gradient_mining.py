"""Gradient Mining — Signal-to-Noise amplification for LoRA training.

Tracks per-parameter gradient statistics (EMA of mean and variance) over time.
Parameters with consistent directional signal that's too quiet to drive learning
are selectively amplified. Parameters with noisy, directionless gradients are
left alone or suppressed.

No external reference images, no extra loss term, no second forward/backward pass.
Just listens harder to what the model is already trying to learn.

Usage:
    miner = GradientMiner(ema_decay=0.99, amplify_scale=2.0, min_snr=0.1)
    ...
    # After backward, before optimizer.step():
    stats = miner.amplify_gradients(network)
    optimizer.step()
"""

import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)


class GradientMiner:
    """Amplifies suppressed learning signal based on per-parameter gradient SNR.

    Tracks exponential moving averages of gradient magnitude (signal) and
    gradient variance (noise) for each parameter. Parameters with high
    signal-to-noise ratio (consistent direction, low variance) get amplified.
    Parameters with low SNR (noisy, directionless) are left unchanged.
    """

    def __init__(
        self,
        ema_decay: float = 0.99,
        amplify_scale: float = 2.0,
        min_snr: float = 0.1,
    ):
        """
        Args:
            ema_decay: EMA smoothing factor for gradient stats (0.99 = slow adaptation,
                      0.9 = fast adaptation). Higher = more history, more stable detection.
            amplify_scale: Maximum amplification factor for high-SNR parameters.
                          1.0 = no amplification, 2.0 = up to 2x, etc.
            min_snr: Minimum SNR threshold below which no amplification is applied.
                    Prevents amplifying pure noise.
        """
        self.ema_decay = ema_decay
        self.amplify_scale = amplify_scale
        self.min_snr = min_snr

        # Per-parameter EMA stats — populated on first step
        self._ema_mean: Dict[str, torch.Tensor] = {}   # EMA of grad (signed — tracks direction)
        self._ema_sq: Dict[str, torch.Tensor] = {}     # EMA of grad^2 (tracks variance)
        self._step_count = 0

        # Stats for logging
        self.last_amplified_ratio = 0.0
        self.last_avg_snr = 0.0
        self.last_avg_boost = 0.0

    def amplify_gradients(self, network: torch.nn.Module) -> Dict[str, float]:
        """Analyze and amplify gradients in-place based on accumulated SNR stats.

        Call this AFTER backward() and BEFORE optimizer.step().

        Args:
            network: The LoRA network with .grad populated on parameters.

        Returns:
            Dict with mining stats for logging.
        """
        self._step_count += 1
        total_params = 0
        amplified_params = 0
        snr_sum = 0.0
        boost_sum = 0.0

        for name, param in network.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            grad = param.grad
            total_params += 1

            # Update EMA stats
            if name not in self._ema_mean:
                # First step — initialise from current gradient
                self._ema_mean[name] = grad.detach().clone()
                self._ema_sq[name] = (grad.detach() ** 2).clone()
                continue  # Don't amplify on first step — no history yet

            # EMA update: mean tracks signed direction, sq tracks magnitude
            self._ema_mean[name].mul_(self.ema_decay).add_(grad.detach(), alpha=1.0 - self.ema_decay)
            self._ema_sq[name].mul_(self.ema_decay).add_(grad.detach() ** 2, alpha=1.0 - self.ema_decay)

            # Compute per-element signal-to-noise ratio
            signal = self._ema_mean[name].abs()       # consistent direction = high
            variance = self._ema_sq[name] - self._ema_mean[name] ** 2
            noise = variance.clamp(min=0).sqrt()       # std dev of gradient

            snr = signal / (noise + 1e-8)

            # Average SNR for this parameter (for logging)
            param_snr = snr.mean().item()
            snr_sum += param_snr

            # Compute amplification factor: scales from 1.0 to amplify_scale
            # based on SNR. High SNR = amplify more. Low SNR = leave alone.
            # Using tanh to smoothly saturate at amplify_scale.
            boost = 1.0 + (self.amplify_scale - 1.0) * torch.tanh(snr - self.min_snr).clamp(min=0)

            # Only count as "amplified" if boost is meaningfully above 1.0
            n_boosted = (boost > 1.05).sum().item()
            if n_boosted > 0:
                amplified_params += 1
                boost_sum += boost.mean().item()

            # Apply amplification in-place
            param.grad.mul_(boost)

        # Update stats
        self.last_amplified_ratio = amplified_params / max(total_params, 1)
        self.last_avg_snr = snr_sum / max(total_params, 1)
        self.last_avg_boost = boost_sum / max(amplified_params, 1) if amplified_params > 0 else 1.0

        return {
            "amplified": amplified_params,
            "total": total_params,
            "ratio": self.last_amplified_ratio,
            "avg_snr": self.last_avg_snr,
            "avg_boost": self.last_avg_boost,
        }
