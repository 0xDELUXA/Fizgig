"""Gradient Mining — Directional filtering + per-block SNR weighting for LoRA training.

Tracks per-parameter gradient statistics (EMA of direction and variance) over time.
Each step:
1. Filters gradients directionally — keeps only the component that agrees with
   historical learning direction, discards the noisy perpendicular component.
2. Amplifies based on per-element SNR — consistent signal gets boosted.
3. Weights by per-block SNR — transformer blocks that are consistently learning
   get an additional boost; blocks with scattered/noisy gradients are dampened.

No external reference images, no extra loss term, no second forward/backward pass.

Usage:
    miner = GradientMiner(ema_decay=0.95, amplify_scale=2.0)
    ...
    # After backward, before optimizer.step():
    stats = miner.amplify_gradients(network)
    optimizer.step()
"""

import re
import logging
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Regex to extract block name from LoRA parameter name
# e.g. "lora_unet_double_blocks_3_img_attn_qkv.lora_down.weight" → "double_3"
# e.g. "lora_unet_single_blocks_15_linear1.lora_up.weight" → "single_15"
_BLOCK_RE = re.compile(r"(double_blocks|single_blocks)[._](\d+)")


def _extract_block_name(param_name: str) -> Optional[str]:
    """Extract block identifier from a LoRA parameter name."""
    m = _BLOCK_RE.search(param_name)
    if m:
        block_type = "double" if "double" in m.group(1) else "single"
        return f"{block_type}_{m.group(2)}"
    return None


class GradientMiner:
    """Directional gradient filtering with per-block SNR weighting.

    Combines three mechanisms:
    1. Element-wise directional filter — zeros out gradient elements that
       disagree with historical EMA direction (sign mismatch = noise).
    2. Per-element SNR amplification — consistent-direction elements get
       boosted proportional to their signal-to-noise ratio.
    3. Per-block weighting — transformer blocks with higher average SNR
       get an additional multiplier; noisy blocks are dampened.
    """

    def __init__(
        self,
        ema_decay: float = 0.95,
        amplify_scale: float = 2.0,
        min_snr: float = 0.1,
        auto_threshold: bool = False,
    ):
        """
        Args:
            ema_decay: EMA smoothing factor (0.95 = fast, 0.99 = slow).
            amplify_scale: Maximum amplification for high-SNR elements.
            min_snr: Minimum SNR threshold. Ignored when auto_threshold is True.
            auto_threshold: Auto-detect noise floor from SNR distribution variance.
        """
        self.ema_decay = ema_decay
        self.amplify_scale = amplify_scale
        self.min_snr = min_snr
        self.auto_threshold = auto_threshold

        # Per-parameter EMA stats
        self._ema_mean: Dict[str, torch.Tensor] = {}
        self._ema_sq: Dict[str, torch.Tensor] = {}
        self._step_count = 0

        # Stats for logging
        self.last_avg_snr = 0.0
        self.last_avg_boost = 0.0
        self.last_threshold = min_snr
        self.last_block_range = (1.0, 1.0)
        self.last_filtered_ratio = 0.0

    def amplify_gradients(self, network: torch.nn.Module) -> Dict[str, float]:
        """Filter and amplify gradients in-place.

        Call AFTER backward() and BEFORE optimizer.step().
        """
        self._step_count += 1
        total_params = 0
        snr_sum = 0.0
        boost_sum = 0.0
        boost_count = 0
        total_elements = 0
        filtered_elements = 0

        # ── Pass 1: Update EMA, compute per-param SNR, collect block stats ──
        param_data = {}  # name -> (snr_tensor, snr_mean, block_name)
        block_snr_accum = {}  # block_name -> [snr_mean, ...]

        for name, param in network.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            grad = param.grad
            total_params += 1

            if name not in self._ema_mean:
                self._ema_mean[name] = grad.detach().clone()
                self._ema_sq[name] = (grad.detach() ** 2).clone()
                continue

            # EMA update
            self._ema_mean[name].mul_(self.ema_decay).add_(grad.detach(), alpha=1.0 - self.ema_decay)
            self._ema_sq[name].mul_(self.ema_decay).add_(grad.detach() ** 2, alpha=1.0 - self.ema_decay)

            # Per-element SNR
            signal = self._ema_mean[name].abs()
            variance = (self._ema_sq[name] - self._ema_mean[name] ** 2).clamp(min=0)
            noise = variance.sqrt()
            snr = signal / (noise + 1e-8)

            snr_mean = snr.mean().item()
            snr_sum += snr_mean

            block_name = _extract_block_name(name)
            param_data[name] = (snr, snr_mean, block_name)

            if block_name is not None:
                block_snr_accum.setdefault(block_name, []).append(snr_mean)

        # ── Auto-threshold ──
        if self.auto_threshold and len(param_data) > 1:
            all_snr = [d[1] for d in param_data.values()]
            mean_snr = sum(all_snr) / len(all_snr)
            var_snr = sum((s - mean_snr) ** 2 for s in all_snr) / len(all_snr)
            std_snr = var_snr ** 0.5
            effective_threshold = max(mean_snr - 0.5 * std_snr, 0.001)
        else:
            effective_threshold = self.min_snr

        # ── Per-block weighting ──
        block_weights = {}
        if block_snr_accum:
            block_avg = {b: sum(s) / len(s) for b, s in block_snr_accum.items()}
            overall_avg = sum(block_avg.values()) / max(len(block_avg), 1)
            if overall_avg > 1e-8:
                block_weights = {
                    b: max(0.5, min(2.0, v / overall_avg))
                    for b, v in block_avg.items()
                }

        min_bw = min(block_weights.values()) if block_weights else 1.0
        max_bw = max(block_weights.values()) if block_weights else 1.0

        # ── Pass 2: Directional filter + SNR amplify + block weight ──
        for name, param in network.named_parameters():
            if name not in param_data:
                continue

            snr, snr_mean, block_name = param_data[name]
            grad = param.grad

            # 1. Directional filter: zero out elements that disagree with EMA direction
            ema_sign = self._ema_mean[name].sign()
            grad_sign = grad.sign()
            agreement = (grad_sign == ema_sign).float()
            filtered_grad = grad * agreement

            n_elements = grad.numel()
            n_filtered = n_elements - int(agreement.sum().item())
            total_elements += n_elements
            filtered_elements += n_filtered

            # 2. Per-element SNR boost
            boost = 1.0 + (self.amplify_scale - 1.0) * torch.tanh(snr - effective_threshold).clamp(min=0)

            # 3. Per-block weight
            bw = block_weights.get(block_name, 1.0) if block_name else 1.0

            # Combined
            param.grad = filtered_grad * boost * bw

            avg_boost = (boost * bw).mean().item()
            if avg_boost > 1.05:
                boost_count += 1
                boost_sum += avg_boost

        # ── Update stats ──
        self.last_avg_snr = snr_sum / max(total_params, 1)
        self.last_avg_boost = boost_sum / max(boost_count, 1) if boost_count > 0 else 1.0
        self.last_threshold = effective_threshold
        self.last_block_range = (round(min_bw, 2), round(max_bw, 2))
        self.last_filtered_ratio = filtered_elements / max(total_elements, 1)

        return {
            "avg_snr": self.last_avg_snr,
            "avg_boost": self.last_avg_boost,
            "threshold": self.last_threshold,
            "blk_min": self.last_block_range[0],
            "blk_max": self.last_block_range[1],
            "filtered": self.last_filtered_ratio,
        }
