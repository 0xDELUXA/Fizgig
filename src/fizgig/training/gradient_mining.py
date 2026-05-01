"""Gradient Mining — Live directional filtering with face crop separation.

Buckets track gradient direction patterns per parameter as training progresses.
Each gradient is split into parallel (aligned with bucket direction) and
orthogonal (noise) components. The parallel signal is preserved, orthogonal
is suppressed to 0.2x. No amplification — only directional cleanup.

Face crop separation routes face crop gradients to a dedicated bucket pool,
preventing facial feature gradients from competing with composition/style.

Usage:
    miner = GradientMiner(face_separation=True)
    ...
    stats = miner.amplify_gradients(network, is_face_crop=is_face)
    optimizer.step()
"""

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
    """Live directional filtering with face crop separation."""

    def __init__(
        self,
        ema_decay: float = 0.9,
        bucket_threshold: float = 0.1,
        face_separation: bool = False,
    ):
        self.ema_decay = ema_decay
        self.bucket_threshold = bucket_threshold
        self.face_separation = face_separation

        # Per-parameter bucket storage: name -> [(ema_dir, ema_sq, hit_count), ...]
        self._buckets: Dict[str, List[List]] = {}
        self._buckets_face: Dict[str, List[List]] = {}
        self._step_count = 0

        # Bucket cap: 8 per pool when separated, 12 when not
        self._bucket_cap = 8 if face_separation else 12

        # Cached block name lookups
        self._block_name_cache: Dict[str, Optional[str]] = {}

        # Stats
        self.last_avg_agreement = 0.0
        self.last_avg_buckets = 1.0

    def _get_active_pool(self, is_face_crop: bool):
        """Return buckets_dict for the active pool."""
        if self.face_separation and is_face_crop:
            return self._buckets_face
        return self._buckets

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

    def amplify_gradients(self, network: torch.nn.Module, is_face_crop: bool = False) -> Dict[str, float]:
        """Track bucket directions and apply directional filtering to gradients."""
        self._step_count += 1
        active_buckets = self._get_active_pool(is_face_crop)
        effective_ema = self.ema_decay

        agreement_sum = 0.0
        bucket_count_sum = 0
        n_filtered = 0

        for name, param in network.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            grad = param.grad.detach()

            if name not in self._block_name_cache:
                self._block_name_cache[name] = _extract_block_name(name)

            # Initialise buckets for new parameters
            if name not in active_buckets:
                active_buckets[name] = [[grad.clone(), (grad ** 2).clone(), 1]]
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
            else:
                # Create new bucket, cap per pool with merge-to-make-room
                if len(param_buckets) >= self._bucket_cap:
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
                    hi, hj = param_buckets[merge_i][2], param_buckets[merge_j][2]
                    tot = hi + hj
                    wi, wj = hi / max(tot, 1), hj / max(tot, 1)
                    param_buckets[merge_i][0] = param_buckets[merge_i][0] * wi + param_buckets[merge_j][0] * wj
                    param_buckets[merge_i][1] = param_buckets[merge_i][1] * wi + param_buckets[merge_j][1] * wj
                    param_buckets[merge_i][2] = tot
                    param_buckets.pop(merge_j)
                param_buckets.append([grad.clone(), (grad ** 2).clone(), 1])
                best_idx = len(param_buckets) - 1

            # Directional filtering against matched bucket
            ema_dir = param_buckets[best_idx][0]
            ema_norm = ema_dir.norm()
            if ema_norm < 1e-10:
                continue

            direction = ema_dir / ema_norm
            dot = (param.grad * direction).sum()
            parallel = dot * direction
            orthogonal = param.grad - parallel

            grad_norm = param.grad.norm()
            if grad_norm > 1e-10:
                cos_sim = (dot / grad_norm).clamp(-1.0, 1.0).item()
            else:
                cos_sim = 0.0
            agreement = (cos_sim + 1.0) * 0.5
            agreement_sum += agreement
            n_filtered += 1

            # Filter: keep parallel at agreement strength, suppress orthogonal to 0.2x
            param.grad = parallel * agreement + orthogonal * 0.2

        # Update stats
        self.last_avg_agreement = agreement_sum / max(n_filtered, 1)
        self.last_avg_buckets = bucket_count_sum / max(n_filtered, 1)

        return {
            "agree": self.last_avg_agreement,
            "bkts": self.last_avg_buckets,
            "face": is_face_crop,
        }
