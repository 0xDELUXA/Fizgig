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
        auto_threshold: bool = False,
    ):
        """
        Args:
            ema_decay: EMA smoothing factor for gradient stats (0.99 = slow adaptation,
                      0.9 = fast adaptation). Higher = more history, more stable detection.
            amplify_scale: Maximum amplification factor for high-SNR parameters.
                          1.0 = no amplification, 2.0 = up to 2x, etc.
            min_snr: Minimum SNR threshold below which no amplification is applied.
                    Prevents amplifying pure noise. Ignored when auto_threshold is True.
            auto_threshold: If True, automatically detect the noise floor from the
                          variance of the SNR distribution across parameters. Replaces
                          the fixed min_snr with an adaptive threshold each step.
        """
        self.ema_decay = ema_decay
        self.amplify_scale = amplify_scale
        self.min_snr = min_snr
        self.auto_threshold = auto_threshold

        # Per-parameter EMA stats — populated on first step
        self._ema_mean: Dict[str, torch.Tensor] = {}   # EMA of grad (signed — tracks direction)
        self._ema_sq: Dict[str, torch.Tensor] = {}     # EMA of grad^2 (tracks variance)
        self._step_count = 0

        # Stats for logging
        self.last_amplified_ratio = 0.0
        self.last_avg_snr = 0.0
        self.last_avg_boost = 0.0
        self.last_threshold = min_snr
        self._last_auto_threshold = min_snr

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

        # Pass 1: Update EMA stats and collect per-parameter SNR values
        param_snr_map = {}  # name -> (snr_tensor, param_snr_mean)
        for name, param in network.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            grad = param.grad
            total_params += 1

            # Update EMA stats
            if name not in self._ema_mean:
                self._ema_mean[name] = grad.detach().clone()
                self._ema_sq[name] = (grad.detach() ** 2).clone()
                continue  # No history yet — skip

            self._ema_mean[name].mul_(self.ema_decay).add_(grad.detach(), alpha=1.0 - self.ema_decay)
            self._ema_sq[name].mul_(self.ema_decay).add_(grad.detach() ** 2, alpha=1.0 - self.ema_decay)

            signal = self._ema_mean[name].abs()
            variance = self._ema_sq[name] - self._ema_mean[name] ** 2
            noise = variance.clamp(min=0).sqrt()
            snr = signal / (noise + 1e-8)

            param_snr_mean = snr.mean().item()
            snr_sum += param_snr_mean
            param_snr_map[name] = (snr, param_snr_mean)

        # Auto-threshold: detect noise floor from SNR distribution variance
        if self.auto_threshold and len(param_snr_map) > 1:
            all_snr_means = [v[1] for v in param_snr_map.values()]
            snr_mean_of_means = sum(all_snr_means) / len(all_snr_means)
            snr_var = sum((s - snr_mean_of_means) ** 2 for s in all_snr_means) / len(all_snr_means)
            snr_std = snr_var ** 0.5

            # Threshold = mean - 0.5*std: parameters below this are in the noise floor
            # If distribution is flat (low std), threshold rises → less amplification
            # If distribution has structure (high std), threshold drops → more amplification
            effective_threshold = max(snr_mean_of_means - 0.5 * snr_std, 0.001)
            self._last_auto_threshold = effective_threshold
        else:
            effective_threshold = self.min_snr

        # Pass 2: Apply amplification using the threshold
        for name, param in network.named_parameters():
            if name not in param_snr_map:
                continue

            snr, param_snr_mean = param_snr_map[name]

            boost = 1.0 + (self.amplify_scale - 1.0) * torch.tanh(snr - effective_threshold).clamp(min=0)

            n_boosted = (boost > 1.05).sum().item()
            if n_boosted > 0:
                amplified_params += 1
                boost_sum += boost.mean().item()

            param.grad.mul_(boost)

        # Update stats
        self.last_amplified_ratio = amplified_params / max(total_params, 1)
        self.last_avg_snr = snr_sum / max(total_params, 1)
        self.last_avg_boost = boost_sum / max(amplified_params, 1) if amplified_params > 0 else 1.0
        self.last_threshold = effective_threshold

        return {
            "amplified": amplified_params,
            "total": total_params,
            "ratio": self.last_amplified_ratio,
            "avg_snr": self.last_avg_snr,
            "avg_boost": self.last_avg_boost,
            "threshold": self.last_threshold,
        }
