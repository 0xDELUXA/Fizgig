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
    """Append-only per-image loss recorder. No-op unless the env flag is set (or force=True)."""

    def __init__(self, output_dir: str, ema_beta: float = 0.9, force: bool = False):
        self.enabled = force or is_enabled()
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


class PerImageLossWatch:
    """Online per-image difficulty watcher — the actionable layer on top of the passive logger.

    Two GUI-toggleable behaviours share this one class:
      * problem-image detection (`write_jsonl` / always-on analytics): at each epoch boundary,
        classifies every image as STUCK (high residual, not descending — likely outlier/mislabel),
        LEARNING (high but descending — hard-but-good), or easy, logs stuck images to the console,
        and writes <output_dir>/loss_log/problem_images.json.
      * per-image adaptive LR (`apply_lr=True`): additionally emits a per-image loss multiplier —
        throttle STUCK images (default x0.5) so one bad image can't keep yanking the weights, gently
        ease off easy/learned ones (default x0.9), leave hard-but-learning alone. At batch size 1,
        scaling the step's loss IS a per-image LR. Guardrails: no action during the warmup epochs
        (the trend needs data), multipliers only ever reduce (never boost), and batch size > 1
        disables scaling entirely (a batch mean isn't a per-image signal).

    Normalization matches the validated offline analyzer: residuals are recomputed at every epoch
    boundary against the CURRENT per-timestep-bucket means over the whole run so far — never the
    order-dependent live running mean (first sample per bucket would always read as residual 0).
    Raw records are kept in memory; a few thousand (key, epoch, bucket, loss) tuples is trivial.
    """

    def __init__(self, output_dir: str, *, apply_lr: bool = False, write_jsonl: bool = False,
                 warmup_epochs: int = 2, window: int = 5,
                 throttle_mult: float = 0.5, easy_mult: float = 0.9,
                 hi_q: float = 0.66, lo_q: float = 0.33,
                 persist_on: int = 2, persist_off: int = 3,
                 improve_frac: float = 0.12, improve_floor: float = 0.02,
                 suspect_mult: float = 0.7, easy_from_epoch: int = 3,
                 exhausted_mult: float = 0.6, exhaust_drop_frac: float = 0.3, exhaust_on: int = 2,
                 stuck_floor: float = 0.1, escalate_every: int = 2,
                 dataset_dir: str = None, caption_ext: str = ".txt"):
        self.apply_lr = apply_lr
        self.warmup_epochs = warmup_epochs
        self.window = window
        self.throttle_mult = throttle_mult
        self.easy_mult = easy_mult
        self.hi_q = hi_q
        self.lo_q = lo_q
        self.persist_on = persist_on      # consecutive stuck votes to CONFIRM stuck
        self.persist_off = persist_off    # consecutive clear votes to RELEASE stuck
        # "Improving" test: split the last trend_window epochs into two halves and require the
        # recent half's mean residual to sit below the older half's by BOTH a fraction of the
        # excess AND ~1 standard error (estimated from that image's own residual scatter). With
        # one loss sample per image per epoch, slope signs and point-to-point drops both flap
        # inside the noise (validated on synthetic runs) — half-window means with a data-driven
        # noise bar is what actually separates a learner from a stuck outlier.
        self.improve_frac = improve_frac
        self.improve_floor = improve_floor
        self.trend_window = 8             # epochs; halves of 4 vs 4
        # Early-suspicion tier: caption-contradiction images are separable by MAGNITUDE from the
        # first epochs (extreme residual, not merely high), long before a trend exists. A robust
        # outlier (median + 1.5*IQR) for 2 consecutive boundaries — or a wild one (3*IQR) once —
        # gets a provisional, mild suspect_mult while the trend machinery gathers evidence.
        # Rationale: modern runs form identity in epochs 1-6 (real likeness by epoch 3 on clean
        # data), so waiting for trend confirmation acts after the damage window has closed.
        self.suspect_mult = suspect_mult
        self.easy_from_epoch = easy_from_epoch  # easy ease-off needs a slightly steadier baseline
        # "Exhausted" tier: an image that PROVED a good run (residual dropped >= exhaust_drop_frac
        # of its early baseline) and then plateaued while still above-average difficulty. Its
        # caption is fine — the model has mined what it can — so further full-LR passes are mostly
        # overbake pressure. Gets exhausted_mult, and its improvement history SUPPRESSES the stuck/
        # suspect paths (a plateaued good-runner is not an outlier, whatever its current level).
        self.exhausted_mult = exhausted_mult
        self.exhaust_drop_frac = exhaust_drop_frac
        self.exhaust_on = exhaust_on
        # Escalating stuck throttle: staying confirmed IS accumulating evidence, so the penalty
        # deepens — throttle_mult on confirmation, halved every escalate_every further confirmed
        # epochs, floored at stuck_floor. A flat x0.5 forever still leaks half-strength poison all
        # run; near-zero from day one would deny a false conviction the gradient it needs to prove
        # itself and win release. Caption fixes reset history -> straight back to x1.0.
        self.stuck_floor = stuck_floor
        self.escalate_every = escalate_every
        self.output_dir = output_dir

        self._records: list[tuple[str, int, int, float]] = []   # (key, epoch, bucket, loss)
        self._bsum = [0.0] * _N_BUCKETS
        self._bcnt = [0] * _N_BUCKETS
        self._mult: dict[str, float] = {}
        self.verdicts: dict[str, str] = {}
        self._epochs_seen: set[int] = set()
        self._batched_warned = False
        self._batched = False
        # Persistence state: a single noisy epoch must not flip a verdict. Per-key counters of
        # consecutive stuck / clear votes, plus the confirmed set (drives warnings + throttling).
        self._stuck_votes: dict[str, int] = {}
        self._clear_votes: dict[str, int] = {}
        self._suspect_votes: dict[str, int] = {}
        self._exhaust_votes: dict[str, int] = {}
        self._stuck_epochs: dict[str, int] = {}   # consecutive epochs confirmed (drives escalation)
        self._confirmed_stuck: set[str] = set()
        self._last_reported_stuck: set[str] = set()
        # Keys whose benefit of the doubt is spent (e.g. two failed AI recaptions): re-confirming
        # stuck EXCLUDES them from training entirely for the rest of the run — the trainer skips
        # their steps (no gradient, no loss recorded, so avr_loss stops carrying their permanent
        # error term). One-way, except reset_key (a manual caption edit re-admits the image).
        self._incorrigible: set[str] = set()
        self._excluded: set[str] = set()

        # Persistent exclusions: fizgig_excluded.json lives IN the dataset folder (exclusions are
        # dataset knowledge — they travel with the images, across runs). Each entry snapshots the
        # caption at exclusion time; if a later run finds the .txt changed (user fixed it offline),
        # the entry auto-prunes and the image is re-admitted. Mid-run caption edits un-exclude via
        # reset_key, which also removes the entry.
        self.dataset_dir = dataset_dir
        self.caption_ext = caption_ext
        self._excl_file = (os.path.join(dataset_dir, "fizgig_excluded.json")
                           if dataset_dir and os.path.isdir(dataset_dir) else None)
        self._excl_data: dict[str, dict] = {}
        self._load_persistent_exclusions()

        # The JSONL logger stays the offline source of truth; force it on when the detection
        # toggle asks for it (env var still works on its own).
        self._jsonl = PerImageLossLogger(output_dir, force=write_jsonl)

    # ---- persistent exclusions -------------------------------------------------

    def _current_caption(self, key: str):
        if not self.dataset_dir:
            return None
        p = os.path.join(self.dataset_dir, os.path.basename(key) + self.caption_ext)
        try:
            with open(p, encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return None

    def _load_persistent_exclusions(self) -> None:
        if not self._excl_file or not os.path.exists(self._excl_file):
            return
        try:
            with open(self._excl_file, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            logger.warning("[loss-watch] could not read fizgig_excluded.json — ignoring")
            return
        pruned = []
        for key, entry in dict(data).items():
            cur = self._current_caption(key)
            if cur is not None and entry.get("caption") is not None and cur != entry["caption"]:
                # Caption changed since exclusion (fixed offline) — re-admit.
                pruned.append(key)
                del data[key]
            else:
                self._excluded.add(str(key))
                self._incorrigible.add(str(key))
        self._excl_data = data
        if pruned:
            self._write_persistent_exclusions()
            logger.info(f"[loss-watch] re-admitted {len(pruned)} previously-excluded image(s) "
                        f"whose captions changed: " + ", ".join(os.path.basename(k) for k in pruned))
        if self._excluded:
            logger.warning(f"[loss-watch] {len(self._excluded)} image(s) excluded by a previous run "
                           f"(fizgig_excluded.json) — they will be skipped. Edit their captions or "
                           f"delete the file to re-admit them: "
                           + ", ".join(sorted(os.path.basename(k) for k in self._excluded)))

    def _write_persistent_exclusions(self) -> None:
        if not self._excl_file:
            return
        try:
            tmp = self._excl_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._excl_data, f, indent=2)
            os.replace(tmp, self._excl_file)
        except Exception:
            logger.warning("[loss-watch] could not write fizgig_excluded.json", exc_info=True)

    def _record_exclusion(self, key: str, epoch: int) -> None:
        if not self._excl_file:
            return
        import time
        self._excl_data[key] = {"epoch": int(epoch),
                                "date": time.strftime("%Y-%m-%d %H:%M"),
                                "reason": "still stuck after two AI recaptions",
                                "caption": self._current_caption(key)}
        self._write_persistent_exclusions()

    # ---- per-step ------------------------------------------------------------

    @staticmethod
    def _key_of(item_keys) -> str:
        keys = item_keys if isinstance(item_keys, (list, tuple)) else [item_keys]
        keys = [str(k) for k in keys] if keys else ["<unknown>"]
        return keys[0] if len(keys) == 1 else "|".join(keys)

    def multiplier(self, item_keys) -> float:
        """Loss multiplier for the CURRENT step (looked up before observe; updates at epoch ends)."""
        if not self.apply_lr or self._batched:
            return 1.0
        return self._mult.get(self._key_of(item_keys), 1.0)

    def is_excluded(self, item_keys) -> bool:
        """True when this step's image is excluded from training (skip the step entirely)."""
        if self._batched:
            return False
        return self._key_of(item_keys) in self._excluded

    def observe(self, *, epoch: int, step: int, item_keys, timestep: float, loss: float) -> None:
        """Record one step's RAW (unscaled) loss. Never raises into the training loop."""
        try:
            keys = item_keys if isinstance(item_keys, (list, tuple)) else [item_keys]
            if keys is not None and len(keys) > 1:
                self._batched = True
                if self.apply_lr and not self._batched_warned:
                    self._batched_warned = True
                    logger.warning("[loss-watch] batch size > 1 — per-image LR disabled "
                                   "(a batch-mean loss isn't a per-image signal)")
            t = float(timestep)
            b = min(int(min(max(t, 0.0), 0.999999) * _N_BUCKETS), _N_BUCKETS - 1)
            loss = float(loss)
            key = self._key_of(item_keys)
            self._records.append((key, int(epoch), b, loss))
            self._bsum[b] += loss
            self._bcnt[b] += 1
            self._epochs_seen.add(int(epoch))
            self._jsonl.record(epoch=epoch, step=step, item_keys=item_keys, timestep=t, loss=loss)
        except Exception as e:
            logger.warning(f"[loss-watch] observe failed ({e})")

    # ---- per-epoch -----------------------------------------------------------

    @staticmethod
    def _slope(xs, ys) -> float:
        n = len(xs)
        if n < 2:
            return 0.0
        mx = sum(xs) / n
        my = sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs)
        if denom == 0:
            return 0.0
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom

    def epoch_boundary(self, epoch: int) -> dict:
        """Reclassify every image and refresh multipliers. Returns {key: verdict} (empty pre-warmup)."""
        try:
            if not self._records or len(self._epochs_seen) < self.warmup_epochs:
                return {}
            bucket_mean = [(self._bsum[i] / self._bcnt[i]) if self._bcnt[i] else 0.0
                           for i in range(_N_BUCKETS)]
            per_epoch: dict[str, dict[int, list[float]]] = {}
            raw_sum: dict[str, float] = {}
            raw_cnt: dict[str, int] = {}
            for key, ep, b, loss in self._records:
                per_epoch.setdefault(key, {}).setdefault(ep, []).append(loss - bucket_mean[b])
                raw_sum[key] = raw_sum.get(key, 0.0) + loss
                raw_cnt[key] = raw_cnt.get(key, 0) + 1

            stats = {}
            for key, ep_map in per_epoch.items():
                all_eps = sorted(ep_map)
                series = [sum(ep_map[e]) / len(ep_map[e]) for e in all_eps]  # raw per-epoch residual

                # Trend: compare the two halves of the last trend_window epochs. Half-means average
                # away the single-sample-per-epoch noise; the standard error of their difference
                # (from this image's own scatter) gives the noise bar the improve test must beat.
                tw = min(self.trend_window, len(series) - len(series) % 2)
                old_mean = new_mean = se = 0.0
                if tw >= 4:
                    win = series[-tw:]
                    half = tw // 2
                    old_half, new_half = win[:half], win[half:]
                    old_mean = sum(old_half) / half
                    new_mean = sum(new_half) / half
                    var = (sum((v - old_mean) ** 2 for v in old_half)
                           + sum((v - new_mean) ** 2 for v in new_half)) / max(tw - 2, 1)
                    se = (2.0 * var / half) ** 0.5

                res = series[-self.window:]
                eps = all_eps[-self.window:]
                # Early baseline: this image's residual level over its first few epochs — the
                # reference for "had a good run" (exhausted detection).
                nb = min(3, len(series))
                baseline = sum(series[:nb]) / nb
                stats[key] = {"mean_residual": sum(res) / len(res),
                              "slope": self._slope([float(e) for e in eps], res),
                              "first": old_mean, "last": new_mean, "se": se,
                              "trend_epochs": tw,
                              "baseline": baseline,
                              "total_drop": baseline - new_mean if tw >= 4 else 0.0,
                              "mean_loss": raw_sum[key] / raw_cnt[key],
                              "epochs": len(all_eps)}

            ordered = sorted(s["mean_residual"] for s in stats.values())
            hi = ordered[int(self.hi_q * (len(ordered) - 1))]
            lo = ordered[int(self.lo_q * (len(ordered) - 1))]
            # Robust outlier thresholds for the early-suspicion tier (floored at hi so the tier
            # never fires on merely-top-third images even when the IQR is degenerate/tiny).
            n = len(ordered)
            med = ordered[n // 2]
            iqr = ordered[int(0.75 * (n - 1))] - ordered[int(0.25 * (n - 1))]
            ext_hi = max(med + 1.5 * iqr, hi)
            ext_vhi = max(med + 3.0 * iqr, hi)
            n_ep = len(self._epochs_seen)

            # Per-epoch RAW votes -> persistence state machine. One noisy epoch can't flip a
            # verdict: stuck is CONFIRMED only after persist_on consecutive stuck votes, and
            # RELEASED only after persist_off consecutive clear votes. The suspect tier acts
            # earlier on MAGNITUDE alone (mild suspect_mult) while the trend evidence accrues.
            new_mult = {}
            for key, s in stats.items():
                # Already excluded: frozen state — no votes, no releases, skipped by the trainer.
                if key in self._excluded:
                    s["verdict"] = "excluded"
                    s["multiplier"] = 0.0
                    new_mult[key] = 0.0
                    continue
                # Improving = recent half-mean below older half-mean by a real margin (fraction of
                # the excess AND ~1 SE). Too little history -> not improving is unknowable; the
                # persistence gate + warmup keep that from mattering.
                drop = s["first"] - s["last"]
                improving = (s["trend_epochs"] >= 4
                             and drop >= max(self.improve_frac * max(s["first"], 0.0),
                                             self.improve_floor, s["se"]))
                s["improving"] = improving
                # "Good run": this image's residual has dropped substantially from its early
                # baseline — proof the caption works and the model CAN learn it. A plateaued
                # good-runner is mined out, not stuck: its history suppresses the outlier paths.
                good_run = (s["trend_epochs"] >= 4 and s["baseline"] > 0.0
                            and s["total_drop"] >= max(self.exhaust_drop_frac * s["baseline"],
                                                       2.0 * self.improve_floor))
                # An image whose recent residual is <= 0 is no harder than average at the same
                # noise level — by definition not an outlier, whatever its trend. And with under
                # 4 epochs of history (fresh run, or history reset after a caption fix) there is
                # no trend to judge — it can't vote stuck yet.
                votes_stuck = (s["trend_epochs"] >= 4 and s["mean_residual"] >= hi
                               and s["last"] > 0.0 and not improving and not good_run)
                if key in self._confirmed_stuck:
                    self._stuck_epochs[key] = self._stuck_epochs.get(key, 0) + 1
                    self._clear_votes[key] = 0 if votes_stuck else self._clear_votes.get(key, 0) + 1
                    if self._clear_votes[key] >= self.persist_off:
                        self._confirmed_stuck.discard(key)
                        self._clear_votes[key] = 0
                        self._stuck_votes[key] = 0
                        # Tenure survives release ON PURPOSE: a noisy 1-2 epoch release must not
                        # reset the escalation ladder — re-confirmation resumes at depth. A real
                        # rehabilitation never re-confirms, and a caption fix wipes it (reset_key).
                else:
                    self._stuck_votes[key] = self._stuck_votes.get(key, 0) + 1 if votes_stuck else 0
                    if self._stuck_votes[key] >= self.persist_on:
                        self._confirmed_stuck.add(key)
                        self._clear_votes[key] = 0
                        self._stuck_epochs[key] = self._stuck_epochs.get(key, 0) + 1

                # Early-suspicion votes: extreme magnitude, not yet improving. (With <4 trend
                # epochs `improving` is always False, which is exactly right here — magnitude is
                # the only early signal, and release comes via the improve test once a trend forms.)
                extreme = s["mean_residual"] >= ext_hi and not improving and not good_run
                self._suspect_votes[key] = self._suspect_votes.get(key, 0) + 1 if extreme else 0
                suspect = (key not in self._confirmed_stuck
                           and (self._suspect_votes[key] >= 2
                                or (self._suspect_votes[key] >= 1 and s["mean_residual"] >= ext_vhi)))

                # Exhausted votes: proved a good run, now plateaued while still above-average
                # difficulty. Improving again (e.g. rest of the dataset caught up, or a caption
                # tweak) releases it immediately.
                exhaust_now = good_run and not improving and s["last"] > 0.0
                self._exhaust_votes[key] = self._exhaust_votes.get(key, 0) + 1 if exhaust_now else 0
                exhausted = self._exhaust_votes[key] >= self.exhaust_on

                # Release progress for the popup (stuck badge + improving trend = counting down).
                s["release_votes"] = self._clear_votes.get(key, 0) if key in self._confirmed_stuck else 0

                if key in self._confirmed_stuck and key in self._incorrigible:
                    # Benefit of the doubt spent (two failed AI recaptions) AND stuck again:
                    # excluded from training for the rest of the run. The trainer skips its steps
                    # — no gradient, no loss recorded — so avr_loss stops carrying its permanent
                    # error term. Only a manual caption edit (reset_key) re-admits it.
                    self._excluded.add(key)
                    self._confirmed_stuck.discard(key)
                    self._last_reported_stuck.discard(key)
                    self._record_exclusion(key, epoch)   # persists to <dataset>/fizgig_excluded.json
                    logger.warning(f"[loss-watch] epoch {epoch}: {os.path.basename(key)} EXCLUDED "
                                   f"from training — two AI captions couldn't fix it. Edit its "
                                   f"caption to re-admit it, or remove it from the dataset.")
                    verdict, mult = "excluded", 0.0
                elif key in self._confirmed_stuck:
                    # Escalate with tenure: x0.5 -> x0.25 -> x0.125 -> floor. Staying confirmed is
                    # accumulating evidence; the leak shrinks as certainty grows.
                    tenure = self._stuck_epochs.get(key, 1)
                    mult = max(self.stuck_floor,
                               self.throttle_mult * (0.5 ** ((tenure - 1) // self.escalate_every)))
                    verdict = "stuck"
                    s["stuck_epochs"] = tenure
                elif suspect:
                    verdict, mult = "suspect", self.suspect_mult   # provisional early throttle
                elif exhausted:
                    verdict, mult = "exhausted", self.exhausted_mult
                elif votes_stuck:
                    verdict, mult = "watch", 1.0          # suspicious this epoch, not yet confirmed
                elif s["mean_residual"] >= hi:
                    verdict, mult = "learning", 1.0
                elif s["mean_residual"] <= lo and n_ep >= self.easy_from_epoch:
                    verdict, mult = "easy", self.easy_mult
                else:
                    verdict, mult = "mid", 1.0
                s["verdict"] = verdict
                s["multiplier"] = mult if self.apply_lr else 1.0
                new_mult[key] = mult
            self._mult = new_mult
            self.verdicts = {k: s["verdict"] for k, s in stats.items()}

            # Excluded images loaded from fizgig_excluded.json are skipped from step 1 — they have
            # no records, so give them stub report rows or they'd be invisible in the popup.
            for k in self._excluded:
                if k not in stats:
                    stats[k] = {"verdict": "excluded", "multiplier": 0.0, "mean_residual": 0.0,
                                "slope": 0.0, "first": 0.0, "last": 0.0, "se": 0.0,
                                "trend_epochs": 0, "baseline": 0.0, "total_drop": 0.0,
                                "mean_loss": 0.0, "epochs": 0, "improving": False}
                    self.verdicts[k] = "excluded"

            # Warn only when the CONFIRMED set changes — not every epoch.
            if self._confirmed_stuck != self._last_reported_stuck:
                added = self._confirmed_stuck - self._last_reported_stuck
                removed = self._last_reported_stuck - self._confirmed_stuck
                action = f" — throttling LR x{self.throttle_mult}" if (self.apply_lr and not self._batched) else ""
                if added:
                    names = ", ".join(os.path.basename(k) for k in sorted(added))
                    logger.warning(f"[loss-watch] epoch {epoch}: image(s) confirmed STUCK "
                                   f"(persistently hard, not improving — check for bad/mislabeled "
                                   f"data): {names}{action}")
                if removed:
                    names = ", ".join(os.path.basename(k) for k in sorted(removed))
                    logger.info(f"[loss-watch] epoch {epoch}: no longer stuck: {names}")
                self._last_reported_stuck = set(self._confirmed_stuck)

            try:
                d = os.path.join(self.output_dir, "loss_log")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "problem_images.json"), "w", encoding="utf-8") as f:
                    json.dump({"epoch": epoch, "apply_lr": self.apply_lr,
                               "images": {k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                                              for kk, vv in s.items()} for k, s in stats.items()}},
                              f, indent=2)
            except Exception:
                pass
            return self.verdicts
        except Exception as e:
            logger.warning(f"[loss-watch] epoch_boundary failed ({e})")
            return {}

    def mark_incorrigible(self, key: str) -> None:
        """Spend a key's benefit of the doubt: if it re-confirms stuck, it goes straight to the
        stuck_floor (no escalation ladder). Called after the 2nd failed AI recaption; a manual
        caption edit (reset_key) clears it."""
        self._incorrigible.add(str(key))

    def reset_key(self, key: str) -> None:
        """Forget one image's history — used after a live caption fix, since its stuck record
        reflects the OLD caption. It re-enters fresh (needs 4 epochs of new trend before it can
        vote stuck again) and its multiplier returns to 1.0 immediately."""
        key = str(key)
        self._records = [r for r in self._records if r[0] != key]
        for d in (self._stuck_votes, self._clear_votes, self._suspect_votes,
                  self._exhaust_votes, self._stuck_epochs, self._mult):
            d.pop(key, None)
        self._confirmed_stuck.discard(key)
        self._last_reported_stuck.discard(key)
        self._incorrigible.discard(key)
        self._excluded.discard(key)   # a manual caption edit re-admits an excluded image
        if key in self._excl_data:    # ...including from the persistent dataset-folder record
            del self._excl_data[key]
            self._write_persistent_exclusions()
        self.verdicts.pop(key, None)

    def close(self) -> None:
        self._jsonl.close()
