"""Passive per-image loss logger (experiment/per-image-loss-logger).

Records, per training step, the image trained on, the timestep it drew, and the loss — so we can
study whether per-image loss *trajectories* (timestep-normalized) reveal image difficulty, outliers,
or a better training order/weighting. This is PURELY OBSERVATIONAL: it never touches gradients, the
learning rate, sampling, or ordering. It only writes a log.

Why timestep matters: a diffusion step's loss is dominated by the *random timestep* drawn that step
(loss at t≈0.95 is structurally huge vs t≈0.2), so raw per-image loss mostly ranks the dice roll,
not the image. We log the raw loss AND the timestep, plus a convenience residual against a running
per-timestep-bucket mean, so the intrinsic "harder/easier than average at this noise level" signal
can be recovered. The raw JSONL is the source of truth; do the real normalization offline.

Enable by setting the env var FIZGIG_PERIMAGE_LOSS_LOG=1 before launching training (the trainer
subprocess inherits it). Off by default → zero cost. Writes:
  <output_dir>/loss_log/per_image_loss.jsonl   — one JSON line per image-step

True per-image granularity needs batch size 1 (each step = one image), which is the Krea 2 /
typical LoRA default. For B>1 the step loss is the batch mean; we log it against the joined keys
and record the batch size so analysis can skip those.
"""
import json
import logging
import os

logger = logging.getLogger(__name__)

_ENV_FLAG = "FIZGIG_PERIMAGE_LOSS_LOG"
_N_BUCKETS = 20  # timestep buckets over [0, 1] for the running-mean normalization


def is_enabled() -> bool:
    """True when FIZGIG_PERIMAGE_LOSS_LOG is set to a truthy value."""
    return os.environ.get(_ENV_FLAG, "").strip().lower() in ("1", "true", "on", "yes")


class PerImageLossLogger:
    """Append-only per-image loss recorder. No-op unless the env flag is set."""

    def __init__(self, output_dir: str, ema_beta: float = 0.9):
        self.enabled = is_enabled()
        self.ema_beta = ema_beta
        self._f = None
        self._ema: dict[str, float] = {}          # item_key -> EMA of raw loss
        self._bucket_sum = [0.0] * _N_BUCKETS     # running loss sum per timestep bucket
        self._bucket_cnt = [0] * _N_BUCKETS
        self.n_records = 0
        if not self.enabled:
            return
        try:
            d = os.path.join(output_dir, "loss_log")
            os.makedirs(d, exist_ok=True)
            self.path = os.path.join(d, "per_image_loss.jsonl")
            self._f = open(self.path, "a", encoding="utf-8")
            logger.info(f"[loss-log] per-image loss logging ON -> {self.path}")
        except Exception as e:
            logger.warning(f"[loss-log] could not open log ({e}); disabling")
            self.enabled = False
            self._f = None

    def _bucket(self, t: float) -> int:
        b = int(min(max(t, 0.0), 0.999999) * _N_BUCKETS)
        return min(b, _N_BUCKETS - 1)

    def record(self, *, epoch: int, step: int, item_keys, timestep: float, loss: float) -> None:
        """Log one image-step. Silently no-ops when disabled; never raises into the training loop."""
        if not self.enabled or self._f is None:
            return
        try:
            loss = float(loss)
            t = float(timestep)
            keys = item_keys if isinstance(item_keys, (list, tuple)) else [item_keys]
            keys = [str(k) for k in keys] if keys else ["<unknown>"]
            key = keys[0] if len(keys) == 1 else "|".join(keys)

            b = self._bucket(t)
            self._bucket_sum[b] += loss
            self._bucket_cnt[b] += 1
            bmean = self._bucket_sum[b] / max(self._bucket_cnt[b], 1)

            prev = self._ema.get(key)
            ema = loss if prev is None else self.ema_beta * prev + (1.0 - self.ema_beta) * loss
            self._ema[key] = ema

            rec = {
                "epoch": int(epoch), "step": int(step), "key": key,
                "t": round(t, 5), "loss": round(loss, 6),
                "t_bucket": b, "t_bucket_mean": round(bmean, 6),
                "residual": round(loss - bmean, 6),   # loss minus running mean at this noise level
                "ema": round(ema, 6),                 # per-image EMA of raw loss
                "batch": len(keys),                   # >1 => batch-mean, not true per-image
            }
            self._f.write(json.dumps(rec) + "\n")
            self._f.flush()
            self.n_records += 1
        except Exception as e:
            logger.warning(f"[loss-log] record failed ({e})")

    def close(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None
