"""Analyze the passive per-image loss log (experiment/per-image-loss-logger).

Reads <output_dir>/loss_log/per_image_loss.jsonl and, per image, reports the timestep-normalized
difficulty (residual) and — crucially — its TREND over epochs. That trend is the whole question:

  * residual high AND descending  -> "learning"  (hard-but-good: DON'T throttle it)
  * residual high AND flat/rising -> "stuck"     (likely outlier/mislabel: throttle or review)
  * residual low                  -> "easy"      (learned: candidate to down-weight)

Residual = raw loss minus the running mean at that step's timestep bucket, so it strips out the
"which random timestep did I draw" confound and leaves "harder/easier than average at this noise
level". We average residual per (image, epoch) — one sample per epoch at batch size 1 — then fit a
straight line across epochs for the slope.

Usage:
    python -m fizgig.scripts.analyze_loss_log <output_dir_or_jsonl> [--top N]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _slope(xs, ys):
    """Least-squares slope of ys vs xs (per epoch). 0.0 if <2 points or degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def main():
    ap = argparse.ArgumentParser(description="Summarize the per-image loss log")
    ap.add_argument("path", help="output dir containing loss_log/, or the per_image_loss.jsonl itself")
    ap.add_argument("--top", type=int, default=20, help="show the N hardest images (by mean residual)")
    args = ap.parse_args()

    jsonl = args.path
    if os.path.isdir(jsonl):
        jsonl = os.path.join(jsonl, "loss_log", "per_image_loss.jsonl")
    if not os.path.exists(jsonl):
        raise SystemExit(f"No log found at {jsonl} (run training with FIZGIG_PERIMAGE_LOSS_LOG=1)")

    # Read raw records first (loss + timestep are the source of truth). We IGNORE the log's live
    # "residual" field — it's a running mean that's order-dependent (first sample per bucket is
    # always 0). Instead we normalize offline against the FINAL per-timestep-bucket mean over the
    # whole run, which is the correct "harder/easier than average at this noise level" signal.
    N_BUCKETS = 20
    records = []          # (key, epoch, loss, bucket)
    bsum = [0.0] * N_BUCKETS
    bcnt = [0] * N_BUCKETS
    batched = 0
    n = 0
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            n += 1
            if r.get("batch", 1) > 1:
                batched += 1
            t = float(r["t"])
            b = min(int(min(max(t, 0.0), 0.999999) * N_BUCKETS), N_BUCKETS - 1)
            loss = float(r["loss"])
            records.append((r["key"], int(r["epoch"]), loss, b))
            bsum[b] += loss
            bcnt[b] += 1

    if not records:
        raise SystemExit("Log is empty.")

    bucket_mean = [(bsum[i] / bcnt[i]) if bcnt[i] else 0.0 for i in range(N_BUCKETS)]

    # key -> epoch -> [offline residuals]; key -> [raw losses]
    per_epoch = defaultdict(lambda: defaultdict(list))
    raw = defaultdict(list)
    for key, epoch, loss, b in records:
        per_epoch[key][epoch].append(loss - bucket_mean[b])
        raw[key].append(loss)

    rows = []
    for key, ep_map in per_epoch.items():
        eps = sorted(ep_map)
        ep_resid = [sum(ep_map[e]) / len(ep_map[e]) for e in eps]     # mean residual per epoch
        mean_res = sum(ep_resid) / len(ep_resid)
        slope = _slope([float(e) for e in eps], ep_resid)             # residual change per epoch
        rows.append({
            "key": key, "epochs": len(eps), "steps": len(raw[key]),
            "mean_residual": mean_res, "slope": slope,
            "first": ep_resid[0], "last": ep_resid[-1],
            "mean_loss": sum(raw[key]) / len(raw[key]),
        })

    # Verdicts relative to the run's own spread.
    resids = sorted(x["mean_residual"] for x in rows)
    hi = resids[int(0.66 * (len(resids) - 1))]     # top third of difficulty
    for x in rows:
        if x["mean_residual"] >= hi and x["slope"] >= -1e-4:
            x["verdict"] = "STUCK (outlier?)"
        elif x["mean_residual"] >= hi:
            x["verdict"] = "learning (hard)"
        else:
            x["verdict"] = "easy"

    rows.sort(key=lambda x: x["mean_residual"], reverse=True)
    stuck = sum(1 for x in rows if x["verdict"].startswith("STUCK"))
    learning = sum(1 for x in rows if x["verdict"].startswith("learning"))

    print(f"\nPer-image loss log: {jsonl}")
    print(f"{n} records over {len(rows)} images"
          + (f"  ({batched} batched rows B>1 — not true per-image)" if batched else ""))
    print(f"Verdicts: {stuck} stuck  |  {learning} learning-hard  |  {len(rows) - stuck - learning} easy\n")
    print(f"{'image':40s} {'ep':>3s} {'meanRes':>9s} {'slope/ep':>10s} {'first→last':>16s}  verdict")
    print("-" * 100)
    for x in rows[: args.top]:
        name = x["key"] if len(x["key"]) <= 40 else "…" + x["key"][-39:]
        print(f"{name:40s} {x['epochs']:>3d} {x['mean_residual']:>9.4f} {x['slope']:>10.5f} "
              f"{x['first']:>7.4f}→{x['last']:<7.4f}  {x['verdict']}")
    print("\nStuck = high residual that isn't descending (review the image). "
          "Learning = high but improving (leave it). Easy = already low.\n")


if __name__ == "__main__":
    main()
