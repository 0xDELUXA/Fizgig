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
        discovery_filter: float = None,
        bucket_threshold: float = 0.1,
        discovery_epochs: int = 2,
        face_separation: bool = False,
    ):
        self.ema_decay = ema_decay
        self.amplify_scale = amplify_scale
        self.min_snr = min_snr
        self.auto_threshold = auto_threshold
        self.filter_strength = orthogonal_scale
        self.discovery_filter = discovery_filter if discovery_filter is not None else orthogonal_scale
        self.bucket_threshold = bucket_threshold
        self.discovery_epochs = discovery_epochs
        self.face_separation = face_separation

        # Per-parameter bucket storage: name -> [(ema_dir, ema_sq, hit_count), ...]
        self._buckets: Dict[str, List[List]] = {}
        self._buckets_face: Dict[str, List[List]] = {}
        self._step_count = 0

        # Bucket cap: 8 per pool when separated, 12 when not
        self._bucket_cap = 8 if face_separation else 12

        # Discovery state
        self._discovery_complete = False
        self._pruned_directions: Dict[str, List[torch.Tensor]] = {}  # blocked directions
        self._current_epoch = 0  # updated by on_epoch_end
        self._refinement_start_step = 0  # step when refinement began
        self._steps_per_epoch = 0  # inferred at discovery end

        # Per-parameter previous gradient for consistency tracking
        self._prev_grad: Dict[str, torch.Tensor] = {}
        self._prev_grad_face: Dict[str, torch.Tensor] = {}

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

    def _get_active_pool(self, is_face_crop: bool):
        """Return (buckets_dict, prev_grad_dict) for the active pool."""
        if self.face_separation and is_face_crop:
            return self._buckets_face, self._prev_grad_face
        return self._buckets, self._prev_grad

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

    def _merge_similar_buckets(self, merge_threshold: float = 0.7, target_buckets: Dict = None) -> int:
        """Merge buckets with cosine similarity above threshold. Returns total merged."""
        if target_buckets is None:
            target_buckets = self._buckets
        total_merged = 0
        for name, buckets in target_buckets.items():
            if len(buckets) <= 1:
                continue
            merged = True
            while merged:
                merged = False
                for i in range(len(buckets)):
                    if buckets[i] is None:
                        continue
                    ema_i = buckets[i][0].flatten()
                    norm_i = ema_i.norm()
                    if norm_i < 1e-10:
                        continue
                    for j in range(i + 1, len(buckets)):
                        if buckets[j] is None:
                            continue
                        ema_j = buckets[j][0].flatten()
                        norm_j = ema_j.norm()
                        if norm_j < 1e-10:
                            continue
                        sim = (ema_i * ema_j).sum() / (norm_i * norm_j)
                        if sim.item() > merge_threshold:
                            hits_i, hits_j = buckets[i][2], buckets[j][2]
                            total = hits_i + hits_j
                            w_i = hits_i / max(total, 1)
                            w_j = hits_j / max(total, 1)
                            buckets[i][0] = buckets[i][0] * w_i + buckets[j][0] * w_j
                            buckets[i][1] = buckets[i][1] * w_i + buckets[j][1] * w_j
                            buckets[i][2] = total
                            buckets[j] = None
                            total_merged += 1
                            merged = True
                            break
                    if merged:
                        break
                buckets = [b for b in buckets if b is not None]
            target_buckets[name] = buckets
        return total_merged

    def on_epoch_end(self, epoch: int):
        """Called by the trainer at the end of each epoch.
        Triggers discovery→refinement transition at the right epoch."""
        self._current_epoch = epoch
        if self._discovery_complete:
            return
        if epoch < self.discovery_epochs:
            return

        # End of discovery — merge similar, then prune weak, then lock
        stats_main = self._finalize_pool(self._buckets)
        stats_face = self._finalize_pool(self._buckets_face) if self.face_separation else None

        self._discovery_complete = True
        self._refinement_start_step = self._step_count
        self._steps_per_epoch = max(self._step_count // self.discovery_epochs, 1)

        msg = (f"[gradient_mining] Discovery complete after {epoch} epoch(s): "
               f"main: {stats_main[0]}→{stats_main[1]} merged, {stats_main[2]} pruned → {stats_main[3]} survived")
        if stats_face:
            msg += (f" | face: {stats_face[0]}→{stats_face[1]} merged, "
                    f"{stats_face[2]} pruned → {stats_face[3]} survived")
        logger.info(msg)

    def _finalize_pool(self, pool: Dict) -> Tuple[int, int, int, int]:
        """Merge + prune a bucket pool. Returns (pre_total, merged, pruned, survived)."""
        pre_total = sum(len(b) for b in pool.values())
        merged = self._merge_similar_buckets(merge_threshold=0.85, target_buckets=pool)
        pruned = 0
        survived = 0

        for name, buckets in pool.items():
            if len(buckets) <= 1:
                survived += len(buckets)
                continue

            total_hits = sum(b[2] for b in buckets)
            max_to_prune = int(len(buckets) * 0.3)

            scored = []
            for b in buckets:
                hit_ratio = b[2] / max(total_hits, 1)
                ema = b[0]
                sq = b[1]
                variance = (sq - ema ** 2).clamp(min=0)
                snr = ema.abs() / (variance.sqrt() + 1e-8)
                snr_score = snr.mean().item()
                keep = (hit_ratio >= 0.03) and (snr_score > 0.5)
                scored.append((b, keep, hit_ratio, snr_score))

            survivors = []
            pruned_count = 0
            for b, keep, hit_ratio, snr_score in scored:
                if keep or pruned_count >= max_to_prune:
                    survivors.append(b)
                else:
                    pruned_count += 1
                    pruned += 1

            if not survivors:
                survivors = [max(buckets, key=lambda b: b[2])]

            pool[name] = survivors
            survived += len(survivors)

        return (pre_total, merged, pruned, survived)

    def amplify_gradients(self, network: torch.nn.Module, is_face_crop: bool = False) -> Dict[str, float]:
        """Filter and amplify gradients using multi-bucket direction tracking."""
        self._step_count += 1
        active_buckets, active_prev_grad = self._get_active_pool(is_face_crop)
        effective_ema = self.ema_decay
        self.last_effective_ema = effective_ema

        # Effective filter: discovery value → tween over 1 epoch → refinement value
        if not self._discovery_complete:
            effective_filter = self.discovery_filter
        else:
            steps_since = self._step_count - self._refinement_start_step
            t = min(1.0, steps_since / max(self._steps_per_epoch, 1))
            effective_filter = self.discovery_filter + t * (self.filter_strength - self.discovery_filter)

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
            if name not in active_buckets:
                active_buckets[name] = [[grad.clone(), (grad ** 2).clone(), 1]]
                active_prev_grad[name] = grad.clone()
                continue

            param_buckets = active_buckets[name]
            grad_flat = grad.flatten()
            bucket_count_sum += len(param_buckets)

            # Find best matching bucket
            best_idx, best_sim = self._find_best_bucket(grad_flat, param_buckets)

            if best_sim > self.bucket_threshold:
                # Update matching bucket
                param_buckets[best_idx][0].mul_(effective_ema).add_(grad, alpha=1.0 - effective_ema)
                param_buckets[best_idx][1].mul_(effective_ema).add_(grad ** 2, alpha=1.0 - effective_ema)
                param_buckets[best_idx][2] += 1
            elif not self._discovery_complete:
                # Discovery phase: create new bucket, cap per pool
                if len(param_buckets) >= self._bucket_cap:
                    # Merge the two most similar existing buckets to make room
                    best_merge_sim = -1.0
                    merge_i, merge_j = 0, 1
                    for mi in range(len(param_buckets)):
                        ei = param_buckets[mi][0].flatten()
                        ni = ei.norm()
                        if ni < 1e-10:
                            continue
                        for mj in range(mi + 1, len(param_buckets)):
                            ej = param_buckets[mj][0].flatten()
                            nj = ej.norm()
                            if nj < 1e-10:
                                continue
                            s = (ei * ej).sum() / (ni * nj)
                            if s.item() > best_merge_sim:
                                best_merge_sim = s.item()
                                merge_i, merge_j = mi, mj
                    # Merge j into i
                    hi, hj = param_buckets[merge_i][2], param_buckets[merge_j][2]
                    tot = hi + hj
                    wi, wj = hi / max(tot, 1), hj / max(tot, 1)
                    param_buckets[merge_i][0] = param_buckets[merge_i][0] * wi + param_buckets[merge_j][0] * wj
                    param_buckets[merge_i][1] = param_buckets[merge_i][1] * wi + param_buckets[merge_j][1] * wj
                    param_buckets[merge_i][2] = tot
                    param_buckets.pop(merge_j)
                param_buckets.append([grad.clone(), (grad ** 2).clone(), 1])
                best_idx = len(param_buckets) - 1
            else:
                # Refinement phase: locked, assign to closest
                param_buckets[best_idx][0].mul_(effective_ema).add_(grad, alpha=1.0 - effective_ema)
                param_buckets[best_idx][1].mul_(effective_ema).add_(grad ** 2, alpha=1.0 - effective_ema)
                param_buckets[best_idx][2] += 1

            # SNR from the matched bucket
            bucket_ema = param_buckets[best_idx][0]
            bucket_sq = param_buckets[best_idx][1]
            signal = bucket_ema.abs()
            variance = (bucket_sq - bucket_ema ** 2).clamp(min=0)
            snr = signal / (variance.sqrt() + 1e-8)
            snr_mean = snr.mean().item()
            snr_sum += snr_mean

            # Directional consistency
            if name in active_prev_grad:
                flat_prev = active_prev_grad[name].flatten()
                if grad_flat.norm() > 1e-10 and flat_prev.norm() > 1e-10:
                    consistency = F.cosine_similarity(grad_flat.unsqueeze(0),
                                                      flat_prev.unsqueeze(0)).item()
                    consistency = max(0.0, consistency)
                else:
                    consistency = 0.0
            else:
                consistency = 0.0

            active_prev_grad[name] = grad.clone()

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

        # ── Auto amplify ──
        # Half amplify during discovery, ramp 1.0→2.0 over first refinement epoch
        if not self._discovery_complete:
            base_amplify = self.amplify_scale * 0.5
        else:
            steps_since = self._step_count - self._refinement_start_step
            ramp = min(1.0, steps_since / max(self._steps_per_epoch, 1))
            base_amplify = 1.0 + ramp * 0.3  # 1.0 → 1.3 over first refinement epoch
        effective_amplify = base_amplify * (0.7 + 0.3 * self.last_avg_agreement)
        effective_amplify = max(1.0, effective_amplify)
        self.last_effective_amplify = effective_amplify

        # ── Pass 2: Directional filter using matched bucket ──
        for name, param in network.named_parameters():
            if name not in param_data:
                continue

            snr, snr_mean, consistency, block_name, best_idx = param_data[name]
            grad = param.grad

            # SNR boost
            boost = 1.0 + (effective_amplify - 1.0) * torch.tanh(snr - effective_threshold).clamp(min=0)

            # Block weight (capped at 1.3 to prevent hidden amplification)
            bw = block_weights.get(block_name, 1.0) if block_name else 1.0
            bw = min(1.3, bw)

            # Face separation: boost identity blocks for face crops, reduce for non-face
            if self.face_separation and block_name is not None and block_name.startswith("single_"):
                try:
                    block_idx = int(block_name.split("_")[1])
                    if 1 <= block_idx <= 16:
                        bw *= 1.3 if is_face_crop else 0.8
                except (ValueError, IndexError):
                    pass
            bw = min(1.5, bw)  # cap total bw including identity modifier

            if effective_filter > 0:
                # Directional filtering: split gradient into parallel/orthogonal
                param_buckets = active_buckets[name]
                best_idx = min(best_idx, len(param_buckets) - 1)
                ema_dir = param_buckets[best_idx][0]
                ema_norm = ema_dir.norm()
                if ema_norm < 1e-10:
                    param.grad = grad * boost * bw
                    continue

                direction = ema_dir / ema_norm
                dot = (grad * direction).sum()
                parallel = dot * direction
                orthogonal = grad - parallel

                grad_norm = grad.norm()
                if grad_norm > 1e-10:
                    cos_sim = (dot / grad_norm).clamp(-1.0, 1.0).item()
                else:
                    cos_sim = 0.0
                agreement = (cos_sim + 1.0) * 0.5
                agreement_sum += agreement

                filtered = parallel * boost * agreement + orthogonal * 0.2
                raw_boosted = grad * boost
                param.grad = (effective_filter * filtered + (1.0 - effective_filter) * raw_boosted) * bw
            else:
                # Pure SNR boost — no directional filtering overhead
                param.grad = grad * boost * bw

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
            "face": is_face_crop,
        }
