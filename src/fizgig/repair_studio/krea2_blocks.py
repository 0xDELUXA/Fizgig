"""Krea 2 block layout for the Repair Studio (and Explorer / Royale when they run in Krea 2
mode). Companion to `state.py`'s Klein layout — the `SliderState` dataclass is shared and
model-agnostic; only the block-id set + the per-block regex differ.

Krea 2's full-model LoRA covers (from a real trained LoRA, 264 modules):
- **28 main transformer blocks** — `lora_unet_blocks_0..27_*` (8 linears each = 224)
- **4 txtfusion blocks** — `txtfusion_layerwise_blocks_0/1`, `txtfusion_refiner_blocks_0/1` (32)
- **8 single I/O linears** — `first`, `last_linear`, `tmlp_0/2`, `tproj_1`, `txtfusion_projector`,
  `txtmlp_1/3` (8). These aren't per-block-controllable; they're left at multiplier 1.

So there are **32 per-block sliders**. Unlike Klein there is **no semantic bucket map yet**
(no Identity/Style/Details categories) — these are generic per-block controls, which is exactly
the instrument for *discovering* Krea 2's block semantics.
"""

import re
from typing import List, Set

KREA2_MAIN_BLOCK_COUNT = 28
KREA2_TXTFUSION_IDS = ["txt_lw_0", "txt_lw_1", "txt_rf_0", "txt_rf_1"]

# block_id -> the literal sub-path that uniquely identifies that txtfusion block.
_TXT_REGEX = {
    "txt_lw_0": r"txtfusion_layerwise_blocks_0_",
    "txt_lw_1": r"txtfusion_layerwise_blocks_1_",
    "txt_rf_0": r"txtfusion_refiner_blocks_0_",
    "txt_rf_1": r"txtfusion_refiner_blocks_1_",
}

# For extracting block ids back out of LoRA module names.
_MAIN_RX = re.compile(r"lora_unet_blocks_(\d+)_")
_TXT_RX = re.compile(r"txtfusion_(layerwise|refiner)_blocks_(\d+)_")


def all_block_ids_krea2() -> List[str]:
    """The 32 per-block slider ids: block_0..block_27, then the four txtfusion blocks."""
    return [f"block_{i}" for i in range(KREA2_MAIN_BLOCK_COUNT)] + list(KREA2_TXTFUSION_IDS)


def block_regex_krea2(block_id: str) -> str:
    """Regex matching a LoRA module's `lora_name` for this block (used by
    `set_module_multiplier_by_pattern`, which does `re.search` on the name).

    Main blocks anchor on `lora_unet_blocks_<N>_` — the `lora_unet_` prefix excludes the
    `txtfusion_..._blocks_N` blocks, and the trailing `_` stops `blocks_2_` from matching
    `blocks_20_`. txtfusion blocks use their distinctive `layerwise|refiner` sub-path.
    """
    if block_id in _TXT_REGEX:
        return _TXT_REGEX[block_id]
    if block_id.startswith("block_"):
        idx = block_id.split("_", 1)[1]
        return rf"lora_unet_blocks_{idx}_"
    raise ValueError(f"unknown Krea 2 block id: {block_id!r}")


def extract_block_ids_krea2(network) -> Set[str]:
    """Return the set of block ids that have LoRA modules in this network (so the UI can grey
    out sliders for blocks a given LoRA doesn't touch). Ignores the 8 single I/O linears."""
    ids: Set[str] = set()
    if network is None:
        return ids
    for mod in getattr(network, "unet_loras", []):
        name = mod.lora_name
        t = _TXT_RX.search(name)
        if t:
            kind = "lw" if t.group(1) == "layerwise" else "rf"
            ids.add(f"txt_{kind}_{int(t.group(2))}")
            continue
        m = _MAIN_RX.search(name)
        if m:
            ids.add(f"block_{int(m.group(1))}")
    return ids
