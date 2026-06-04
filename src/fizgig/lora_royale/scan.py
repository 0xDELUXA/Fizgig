"""Discover LoRA checkpoints to compare in LoRA Royale."""

import os
import re
from typing import List, Tuple

# Trainer epoch checkpoints: "<name>-000005.safetensors" (get_epoch_ckpt_name).
_EPOCH_RE = re.compile(r"^(?P<name>.+)-(?P<epoch>\d{6})\.safetensors$")


def scan_checkpoints(folder: str) -> List[Tuple[int, str]]:
    """Find LoRA checkpoints in `folder`, sorted ascending.

    Returns a list of (label, path) where `label` is the epoch number for
    trainer checkpoints, or a 0-based index for arbitrary LoRAs.

    Behaviour / edge cases:
      - Prefers the trainer's epoch files (`<name>-NNNNNN.safetensors`). Adapts
        to however many exist — an early-stopped run just has fewer.
      - If several runs share the folder, picks the run with the MOST
        checkpoints (the main run) so a stray one-off LoRA doesn't derail it.
      - No epoch files at all → falls back to every `.safetensors` (sorted by
        name) treated as a plain sequence (the 'arbitrary LoRAs' case).
      - State directories (`<name>-NNNNNN-state/`) and non-safetensors are ignored.
    """
    if not folder or not os.path.isdir(folder):
        return []

    by_run = {}   # base name -> [(epoch, path)]
    others = []   # non-epoch .safetensors
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    for fn in names:
        if not fn.endswith(".safetensors"):
            continue
        full = os.path.join(folder, fn)
        if not os.path.isfile(full):
            continue
        m = _EPOCH_RE.match(fn)
        if m:
            by_run.setdefault(m.group("name"), []).append((int(m.group("epoch")), full))
        else:
            others.append(full)

    if by_run:
        best = max(by_run.values(), key=len)
        return sorted(best, key=lambda t: t[0])

    return [(i, p) for i, p in enumerate(sorted(others))]


def run_name_for_folder(folder: str) -> str:
    """Best-guess training-run base name in `folder` (the run with the most
    epoch checkpoints), or '' if none. Used for labelling / promote naming."""
    if not folder or not os.path.isdir(folder):
        return ""
    counts = {}
    try:
        names = os.listdir(folder)
    except OSError:
        return ""
    for fn in names:
        m = _EPOCH_RE.match(fn)
        if m:
            counts[m.group("name")] = counts.get(m.group("name"), 0) + 1
    return max(counts, key=counts.get) if counts else ""
