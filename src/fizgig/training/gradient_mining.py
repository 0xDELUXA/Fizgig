"""Gradient Mining — Multi-bucket directional filtering with discovery/refinement phases.

Discovery phase (first N epochs): unlimited bucket creation, the dataset reveals
its natural gradient pattern count. At the end of discovery, weak buckets (< 5%
of hits) are pruned and their directions blocked.

Refinement phase (remaining epochs): bucket structure locked, no new buckets.
All gradients assigned to surviving buckets. Pruned directions are dampened if
they recur. Each pattern gets full amplification within its own tracked direction.

Usage:
    miner = GradientMiner(amplify_scale=8.0, discovery_epochs=1)
    ...
    stats = miner.amplify_gradients(network)
    optimizer.step()
    # At epoch boundary:
    miner.on_epoch_end(epoch_number)
"""

import math
import re
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_BLOCK_RE = re.compile(r"(double_blocks|single_blocks)[._](\d+)")


def _extract_block_name(param_name: str) -> Optional[str]:
    m = _BLOCK_RE.search(param_name)
    if m:
        block_type = "double" if "double" in m.group(1) else "single"
        return f"{block_type}_{m.group(2)}"
    return None


class GradientMiner:
    """Multi-bucket directional filtering with discovery/refinement phases."""

    def __init__(
        self,
        ema_decay: float = 0.9,
        amplify_scale: float = 8.0,
        min_snr: float = 0.001,
        auto_threshold: bool = True,
        orthogonal_scale: float = 0.3,
        bucket_threshold: float = 0.02,
        discovery_epochs: int = 2,
    ):
        self.ema_decay = ema_decay
        self.amplify_scale = amplify_scale
        self.min_snr = min_snr
        self.auto_threshold = auto_threshold
        self.filter_strength = orthogonal_scale
        self.bucket_threshold = bucket_threshold
        self.discovery_epochs = discovery_epochs

        # Per-parameter bucket storage: name -> [(ema_dir, ema_sq, hit_count), ...]
        self._buckets: Dict[str, List[List]] = {}
        self._step_count = 0

        # Discovery state
        self._discovery_complete = False
        self._pruned_directions: Dict[str, List[torch.Tensor]] = {}  # blocked directions
        self._current_epoch = 0  # updated by on_epoch_end

        # Per-parameter previous gradient for consistency tracking
        self._prev_grad: Dict[str, torch.Tensor] = {}

        # Cached block name lookups
        self._block_name_cache: Dict[str, Optional[str]] = {}

        # Stats
        self.last_avg_snr = 0.0
        self.last_avg_boost = 0.0
        self.last_threshold = min_snr
        self.last_block_entropy = 1.0
        self.last_avg_agreement = 0.0
        self.last_agreement_slope = 0.0
        self._prev_avg_agreement = None
        self.last_effective_amplify = amplify_scale
        self.last_effective_ema = ema_decay
        self.last_avg_buckets = 1.0

    def _find_best_bucket(self, grad_flat: torch.Tensor, buckets: list) -> Tuple[int, float]:
        """Find the bucket with highest cosine similarity to this gradient."""
        best_idx = 0
        best_sim = -1.0
        grad_norm = grad_flat.norm()
        if grad_norm < 1e-10:
            return 0, 0.0
        for i, (bucket_ema, _, _) in enumerate(buckets):
            bucket_flat = bucket_ema.flatten()
            bucket_norm = bucket_flat.norm()
            if bucket_norm < 1e-10:
                continue
            sim = (grad_flat * bucket_flat).sum() / (grad_norm * bucket_norm)
            sim = sim.clamp(-1.0, 1.0).item()
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        return best_idx, best_sim

    def on_epoch_end(self, epoch: int):
        """Called by the trainer at the end of each epoch.
        Triggers discovery→refinement transition at the right epoch."""
        self._current_epoch = epoch
        if self._discovery_complete:
            return
        if epoch < self.discovery_epochs:
            return

        # End of discovery — prune and lock
        total_pruned = 0
        total_survived = 0
        for name, buckets in self._buckets.items():
            if len(buckets) <= 1:
                total_survived += len(buckets)
                continue

            total_hits = sum(b[2] for b in buckets)
            max_to_prune = int(len(buckets) * 0.3)  # never prune more than 30%

            # Score each bucket by frequency AND quality
            scored = []
            for b in buckets:
                hit_ratio = b[2] / max(total_hits, 1)
                # SNR score from this bucket's EMA
                ema = b[0]
                sq = b[1]
                variance = (sq - ema ** 2).clamp(min=0)
                snr = ema.abs() / (variance.sqrt() + 1e-8)
                snr_score = snr.mean().item()
                # Survive if frequent enough AND decent quality
                keep = (hit_ratio >= 0.03) and (snr_score > 0.5)
                scored.append((b, keep, hit_ratio, snr_score))

            survivors = []
            pruned_count = 0
            for b, keep, hit_ratio, snr_score in scored:
                if keep or pruned_count >= max_to_prune:
                    survivors.append(b)
                else:
                    pruned_count += 1
                    total_pruned += 1

            if not survivors:
                survivors = [max(buckets, key=lambda b: b[2])]

            self._buckets[name] = survivors
            total_survived += len(survivors)

        self._discovery_complete = True
        logger.info(
            f"[gradient_mining] Discovery complete after {epoch} epoch(s): "
            f"{total_survived} buckets survived, {total_pruned} pruned"
        )

    def amplify_gradients(self, network: torch.nn.Module) -> Dict[str, float]:
        """Filter and amplify gradients using multi-bucket direction tracking."""
        self._step_count += 1
        effective_ema = self.ema_decay
        self.last_effective_ema = effective_ema

        total_params = 0
        snr_sum = 0.0
        boost_sum = 0.0
        boost_count = 0
        agreement_sum = 0.0
        bucket_count_sum = 0

        # ── Pass 1: Update buckets, compute per-param SNR + consistency ──
        param_data = {}
        block_accum = {}

        for name, param in network.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            grad = param.grad.detach()
            total_params += 1

            if name not in self._block_name_cache:
                self._block_name_cache[name] = _extract_block_name(name)
            block_name = self._block_name_cache[name]

            # Initialise buckets for new parameters
            if name not in self._buckets:
                self._buckets[name] = [[grad.clone(), (grad ** 2).clone(), 1]]
                self._prev_grad[name] = grad.clone()
                continue

            buckets = self._buckets[name]
            grad_flat = grad.flatten()
            bucket_count_sum += len(buckets)

            # Find best matching bucket
            best_idx, best_sim = self._find_best_bucket(grad_flat, buckets)

            if best_sim > self.bucket_threshold:
                # Update matching bucket
                buckets[best_idx][0].mul_(effective_ema).add_(grad, alpha=1.0 - effective_ema)
                buckets[best_idx][1].mul_(effective_ema).add_(grad ** 2, alpha=1.0 - effective_ema)
                buckets[best_idx][2] += 1
            elif not self._discovery_complete:
                # Discovery phase: create new bucket (uncapped)
                buckets.append([grad.clone(), (grad ** 2).clone(), 1])
                best_idx = len(buckets) - 1
            else:
                # Refinement phase: locked, assign to closest
                buckets[best_idx][0].mul_(effective_ema).add_(grad, alpha=1.0 - effective_ema)
                buckets[best_idx][1].mul_(effective_ema).add_(grad ** 2, alpha=1.0 - effective_ema)
                buckets[best_idx][2] += 1

            # SNR from the matched bucket
            bucket_ema = buckets[best_idx][0]
            bucket_sq = buckets[best_idx][1]
            signal = bucket_ema.abs()
            variance = (bucket_sq - bucket_ema ** 2).clamp(min=0)
            snr = signal / (variance.sqrt() + 1e-8)
            snr_mean = snr.mean().item()
            snr_sum += snr_mean

            # Directional consistency
            flat_prev = self._prev_grad[name].flatten()
            if grad_flat.norm() > 1e-10 and flat_prev.norm() > 1e-10:
                consistency = F.cosine_similarity(grad_flat.unsqueeze(0),
                                                  flat_prev.unsqueeze(0)).item()
                consistency = max(0.0, consistency)
            else:
                consistency = 0.0

            self._prev_grad[name].copy_(grad)

            param_data[name] = (snr, snr_mean, consistency, block_name, best_idx)

            if block_name is not None:
                block_accum.setdefault(block_name, []).append((snr_mean, consistency))

        # ── Auto-threshold ──
        if len(param_data) > 1:
            all_snr = [d[1] for d in param_data.values()]
            mean_snr = sum(all_snr) / len(all_snr)
            var_snr = sum((s - mean_snr) ** 2 for s in all_snr) / len(all_snr)
            effective_threshold = max(mean_snr - 0.5 * var_snr ** 0.5, 0.001)
        else:
            effective_threshold = 0.001

        # ── Per-block weighting ──
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

        # ── Auto amplify with epoch decay ──
        # Full amplify during discovery, halved each epoch after, floor at 2.0
        if self._current_epoch <= self.discovery_epochs:
            base_amplify = self.amplify_scale
        else:
            epochs_since = self._current_epoch - self.discovery_epochs
            base_amplify = max(2.0, self.amplify_scale / (2.0 ** epochs_since))
        effective_amplify = base_amplify * (0.7 + 0.3 * self.last_avg_agreement)
        effective_amplify = max(1.0, effective_amplify)
        self.last_effective_amplify = effective_amplify

        # ── Pass 2: Directional filter using matched bucket ──
        for name, param in network.named_parameters():
            if name not in param_data:
                continue

            snr, snr_mean, consistency, block_name, best_idx = param_data[name]
            grad = param.grad
            buckets = self._buckets[name]
            best_idx = min(best_idx, len(buckets) - 1)

            # Direction from matched bucket
            ema_dir = buckets[best_idx][0]
            ema_norm = ema_dir.norm()
            if ema_norm < 1e-10:
                continue

            direction = ema_dir / ema_norm

            # Parallel/orthogonal split against matched bucket direction
            dot = (grad * direction).sum()
            parallel = dot * direction
            orthogonal = grad - parallel

            # Agreement against matched bucket
            grad_norm = grad.norm()
            if grad_norm > 1e-10:
                cos_sim = (dot / grad_norm).clamp(-1.0, 1.0).item()
            else:
                cos_sim = 0.0
            agreement = (cos_sim + 1.0) * 0.5
            agreement_sum += agreement

            # SNR boost
            boost = 1.0 + (effective_amplify - 1.0) * torch.tanh(snr - effective_threshold).clamp(min=0)

            # Block weight
            bw = block_weights.get(block_name, 1.0) if block_name else 1.0

            # Blend filtered + raw
            filtered = parallel * boost * agreement + orthogonal * 0.2
            raw_boosted = grad * boost
            param.grad = (self.filter_strength * filtered + (1.0 - self.filter_strength) * raw_boosted) * bw

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
        self.last_avg_buckets = bucket_count_sum / max(n_with_data, 1)

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
            "bkts": self.last_avg_buckets,
        }
