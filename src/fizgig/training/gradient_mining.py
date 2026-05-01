"""Gradient Mining v2 — Observe + Mine.

Phase 1 (Data Gather): One epoch of vanilla training while silently observing
gradient patterns. Buckets build from unmodified gradients — the model trains
normally while the miner maps the gradient landscape.

Phase 2 (Mining): Bucket structure locked from observation. Directional filtering
+ SNR amplification along the pre-observed directions. The model is pushed hard
along known-good gradient directions.

Phase 3 (Normal): Mining complete. Standard training with no gradient modification.
Adaptive LR takes over from here.

Usage:
    miner = GradientMiner(amplify_scale=8.0, mining_epochs=2)
    ...
    # Phase 1: observe only
    stats = miner.amplify_gradients(network, is_face_crop=False)
    # At epoch 1 end:
    miner.start_mining()  # finalize buckets, lock structure
    # Phase 2: mining active
    stats = miner.amplify_gradients(network, is_face_crop=False)
    # After mining_epochs:
    miner.stop_mining()   # mining done
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
    """Observe + Mine gradient routing with face crop separation."""

    # Phases
    PHASE_OBSERVE = "observe"
    PHASE_MINING = "mining"
    PHASE_DONE = "done"

    def __init__(
        self,
        ema_decay: float = 0.9,
        amplify_scale: float = 8.0,
        min_snr: float = 0.001,
        auto_threshold: bool = True,
        filter_strength: float = 0.5,
        bucket_threshold: float = 0.1,
        mining_epochs: int = 2,
        face_separation: bool = False,
    ):
        self.ema_decay = ema_decay
        self.amplify_scale = amplify_scale
        self.min_snr = min_snr
        self.auto_threshold = auto_threshold
        self.filter_strength = filter_strength
        self.bucket_threshold = bucket_threshold
        self.mining_epochs = mining_epochs
        self.face_separation = face_separation

        # Per-parameter bucket storage: name -> [(ema_dir, ema_sq, hit_count), ...]
        self._buckets: Dict[str, List[List]] = {}
        self._buckets_face: Dict[str, List[List]] = {}
        self._step_count = 0

        # Bucket cap: 8 per pool when separated, 12 when not
        self._bucket_cap = 8 if face_separation else 12

        # Phase state
        self.phase = self.PHASE_OBSERVE
        self._mining_epoch_count = 0  # how many mining epochs have run
        self._steps_per_epoch = 0

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

    def start_mining(self):
        """Called at the end of the data-gather epoch. Finalizes buckets and enters mining phase."""
        if self.phase != self.PHASE_OBSERVE:
            return

        stats_main = self._finalize_pool(self._buckets)
        stats_face = self._finalize_pool(self._buckets_face) if self.face_separation else None

        self._steps_per_epoch = max(self._step_count, 1)
        self.phase = self.PHASE_MINING
        self._mining_epoch_count = 0

        msg = (f"[gradient_mining] Data gather complete: "
               f"main: {stats_main[0]}→{stats_main[1]} merged, {stats_main[2]} pruned → {stats_main[3]} survived")
        if stats_face:
            msg += (f" | face: {stats_face[0]}→{stats_face[1]} merged, "
                    f"{stats_face[2]} pruned → {stats_face[3]} survived")
        logger.info(msg)

    def on_mining_epoch_end(self):
        """Called at the end of each mining epoch. Tracks count for phase transition."""
        if self.phase != self.PHASE_MINING:
            return
        self._mining_epoch_count += 1

    def stop_mining(self):
        """Called when all mining epochs are complete. Enters normal training phase."""
        self.phase = self.PHASE_DONE
        logger.info(f"[gradient_mining] Mining complete after {self._mining_epoch_count} epoch(s). Normal training resumes.")

    def amplify_gradients(self, network: torch.nn.Module, is_face_crop: bool = False) -> Dict[str, float]:
        """Process gradients based on current phase.

        OBSERVE: track bucket EMAs, do NOT modify gradients.
        MINING: directional filtering + SNR boost along locked bucket directions.
        DONE: should not be called (trainer skips).
        """
        self._step_count += 1
        active_buckets, active_prev_grad = self._get_active_pool(is_face_crop)
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
            elif self.phase == self.PHASE_OBSERVE:
                # Observe phase: create new bucket, cap per pool
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
                # Mining phase: locked, assign to closest
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

        # ── OBSERVE phase: tracking only, no gradient modification ──
        if self.phase == self.PHASE_OBSERVE:
            n_with_data = len(param_data)
            self.last_avg_snr = snr_sum / max(n_with_data, 1)
            self.last_avg_buckets = bucket_count_sum / max(n_with_data, 1)
            return {
                "avg_snr": self.last_avg_snr,
                "avg_boost": 1.0,
                "threshold": 0,
                "blk_H": 1.0,
                "agree": 0,
                "d_agree": 0,
                "ema": self.last_effective_ema,
                "amp": 0,
                "bkts": self.last_avg_buckets,
                "face": is_face_crop,
            }

        # ── MINING phase: pure directional filtering (no amplification) ──

        effective_filter = self.filter_strength
        self.last_effective_amplify = 1.0

        # ── Pass 2: Directional filter using locked bucket directions ──
        for name, param in network.named_parameters():
            if name not in param_data:
                continue

            snr, snr_mean, consistency, block_name, best_idx = param_data[name]
            grad = param.grad

            # Directional filtering: split gradient into parallel/orthogonal
            param_buckets = active_buckets[name]
            best_idx = min(best_idx, len(param_buckets) - 1)
            ema_dir = param_buckets[best_idx][0]
            ema_norm = ema_dir.norm()
            if ema_norm < 1e-10:
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

            filtered = parallel * agreement + orthogonal * 0.2
            param.grad = effective_filter * filtered + (1.0 - effective_filter) * grad

        # ── Update stats ──
        n_with_data = len(param_data)
        self.last_avg_snr = snr_sum / max(n_with_data, 1)
        self.last_avg_boost = 1.0
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
