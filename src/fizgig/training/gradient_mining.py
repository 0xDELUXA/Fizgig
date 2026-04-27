"""Gradient Mining — Directional filtering + per-block SNR weighting for LoRA training.

Tracks per-parameter gradient statistics (EMA of direction and variance) over time.
Each step:
1. Splits gradients into parallel (aligned with history) and orthogonal (new direction)
   components. Parallel gets amplified, orthogonal is preserved at reduced scale.
2. Weights by soft agreement — gradients aligned with EMA direction get full weight,
   opposing gradients are suppressed but not zeroed (preserves plasticity).
3. Per-block scoring uses SNR * directional consistency — blocks must be both strong
   AND stable to get boosted.

No external reference images, no extra loss term, no second forward/backward pass.

Usage:
    miner = GradientMiner(ema_decay=0.95, amplify_scale=2.0)
    ...
    # After backward, before optimizer.step():
    stats = miner.amplify_gradients(network)
    optimizer.step()
"""

import math
import re
import logging
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Regex to extract block name from LoRA parameter name
_BLOCK_RE = re.compile(r"(double_blocks|single_blocks)[._](\d+)")


def _extract_block_name(param_name: str) -> Optional[str]:
    m = _BLOCK_RE.search(param_name)
    if m:
        block_type = "double" if "double" in m.group(1) else "single"
        return f"{block_type}_{m.group(2)}"
    return None


class GradientMiner:
    """Directional gradient filtering with per-block SNR+consistency weighting."""

    def __init__(
        self,
        ema_decay: float = 0.95,
        amplify_scale: float = 2.0,
        min_snr: float = 0.1,
        auto_threshold: bool = False,
        orthogonal_scale: float = 0.2,
    ):
        self.ema_decay = ema_decay
        self.amplify_scale = amplify_scale
        self.min_snr = min_snr
        self.auto_threshold = auto_threshold
        self.orthogonal_scale = orthogonal_scale

        # Per-parameter EMA stats
        self._ema_mean: Dict[str, torch.Tensor] = {}
        self._ema_sq: Dict[str, torch.Tensor] = {}
        self._step_count = 0

        # Per-parameter previous gradient for consistency tracking
        self._prev_grad: Dict[str, torch.Tensor] = {}

        # Cached block name lookups (parameter names don't change)
        self._block_name_cache: Dict[str, Optional[str]] = {}

        # Stats for logging
        self.last_avg_snr = 0.0
        self.last_avg_boost = 0.0
        self.last_threshold = min_snr
        self.last_block_entropy = 1.0
        self.last_avg_agreement = 0.0
        self.last_agreement_slope = 0.0
        self._prev_avg_agreement = None
        self.last_effective_ema = ema_decay
        self.last_effective_amplify = amplify_scale

    def amplify_gradients(self, network: torch.nn.Module) -> Dict[str, float]:
        """Filter and amplify gradients in-place. Single-pass optimised.

        Call AFTER backward() and BEFORE optimizer.step().
        """
        self._step_count += 1

        # Auto EMA: adapt decay based on agreement
        effective_ema = 0.9 + 0.09 * self.last_avg_agreement
        self.last_effective_ema = effective_ema

        total_params = 0
        snr_sum = 0.0
        boost_sum = 0.0
        boost_count = 0
        agreement_sum = 0.0

        # ── Pass 1: Update EMA, compute per-param SNR + consistency ──
        param_data = {}  # name -> (snr, snr_mean, consistency, block_name)
        block_accum = {}  # block_name -> [(snr_mean, consistency), ...]

        for name, param in network.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            grad = param.grad.detach()
            total_params += 1

            # Cache block name on first encounter
            if name not in self._block_name_cache:
                self._block_name_cache[name] = _extract_block_name(name)
            block_name = self._block_name_cache[name]

            if name not in self._ema_mean:
                self._ema_mean[name] = grad.clone()
                self._ema_sq[name] = (grad ** 2).clone()
                self._prev_grad[name] = grad.clone()
                continue

            # EMA update
            self._ema_mean[name].mul_(effective_ema).add_(grad, alpha=1.0 - effective_ema)
            self._ema_sq[name].mul_(effective_ema).add_(grad ** 2, alpha=1.0 - effective_ema)

            # Per-element SNR
            signal = self._ema_mean[name].abs()
            variance = (self._ema_sq[name] - self._ema_mean[name] ** 2).clamp(min=0)
            snr = signal / (variance.sqrt() + 1e-8)
            snr_mean = snr.mean().item()
            snr_sum += snr_mean

            # Directional consistency: cheap sign-agreement ratio (replaces cosine_similarity)
            consistency = (grad.sign() == self._prev_grad[name].sign()).float().mean().item()

            # Update prev grad in-place (no allocation after first step)
            self._prev_grad[name].copy_(grad)

            param_data[name] = (snr, snr_mean, consistency, block_name)

            if block_name is not None:
                block_accum.setdefault(block_name, []).append((snr_mean, consistency))

        # Threshold locked at 0.001 — tested as the reliable noise floor
        effective_threshold = 0.001

        # ── Per-block weighting: SNR * consistency ──
        block_weights = {}
        if block_accum:
            block_scores = {}
            for b, entries in block_accum.items():
                avg_snr = sum(e[0] for e in entries) / len(entries)
                avg_con = sum(e[1] for e in entries) / len(entries)
                block_scores[b] = avg_snr * (avg_con + 0.1)

            overall_avg = sum(block_scores.values()) / max(len(block_scores), 1)
            if overall_avg > 1e-8:
                block_weights = {
                    b: max(0.5, min(2.0, v / overall_avg))
                    for b, v in block_scores.items()
                }

        # Block entropy
        if block_weights and len(block_weights) > 1:
            bw_vals = list(block_weights.values())
            bw_total = sum(bw_vals)
            bw_probs = [w / bw_total for w in bw_vals]
            entropy = -sum(p * math.log2(p) for p in bw_probs if p > 0)
            max_entropy = math.log2(len(bw_vals))
            block_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
        else:
            block_entropy = 1.0

        # ── Auto amplify: scale amplification by agreement level ──
        # agree=0.6 → full amplify. Higher → push harder. Lower → back off.
        effective_amplify = self.amplify_scale * (self.last_avg_agreement / 0.6)
        effective_amplify = max(1.0, min(8.0, effective_amplify))  # floor 1.0, ceiling 8.0
        self.last_effective_amplify = effective_amplify

        # ── Pass 2: Directional filter + amplify + block weight ──
        for name, param in network.named_parameters():
            if name not in param_data:
                continue

            snr, snr_mean, consistency, block_name = param_data[name]
            grad = param.grad

            # Normalised EMA direction
            ema_dir = self._ema_mean[name]
            ema_norm = ema_dir.norm()
            if ema_norm < 1e-10:
                continue

            direction = ema_dir / ema_norm

            # Parallel/orthogonal split
            dot = (grad * direction).sum()
            parallel = dot * direction
            orthogonal = grad - parallel

            # Agreement from dot product (direction is unit-length, so cos = dot / grad_norm)
            grad_norm = grad.norm()
            if grad_norm > 1e-10:
                cos_sim = (dot / grad_norm).clamp(-1.0, 1.0).item()
            else:
                cos_sim = 0.0
            agreement = (cos_sim + 1.0) * 0.5  # [-1,1] → [0,1]
            agreement_sum += agreement

            # Per-element SNR boost (using auto-scaled amplify)
            boost = 1.0 + (effective_amplify - 1.0) * torch.tanh(snr - effective_threshold).clamp(min=0)

            # Block weight
            bw = block_weights.get(block_name, 1.0) if block_name else 1.0

            # Combined
            param.grad = (parallel * boost * agreement + orthogonal * self.orthogonal_scale) * bw

            avg_boost_val = (boost * bw).mean().item()
            if avg_boost_val > 1.05:
                boost_count += 1
                boost_sum += avg_boost_val

        # ── Update stats ──
        n_with_data = len(param_data)
        self.last_avg_snr = snr_sum / max(n_with_data, 1)
        self.last_avg_boost = boost_sum / max(boost_count, 1) if boost_count > 0 else 1.0
        self.last_threshold = effective_threshold
        self.last_block_entropy = block_entropy
        current_agreement = agreement_sum / max(n_with_data, 1)

        if self._prev_avg_agreement is not None:
            self.last_agreement_slope = current_agreement - self._prev_avg_agreement
        else:
            self.last_agreement_slope = 0.0
        self._prev_avg_agreement = current_agreement
        self.last_avg_agreement = current_agreement

        return {
            "avg_snr": self.last_avg_snr,
            "avg_boost": self.last_avg_boost,
            "threshold": self.last_threshold,
            "blk_H": self.last_block_entropy,
            "agree": self.last_avg_agreement,
            "d_agree": self.last_agreement_slope,
            "ema": self.last_effective_ema,
            "amp": self.last_effective_amplify,
        }
