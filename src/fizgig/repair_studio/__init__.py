"""LoRA Repair Studio — live per-block LoRA tweaking with side-by-side preview.

v1: full-forward preview per slider change (slow but correct).
v2 (planned): activation caching keyed on earliest-changed-block.

Public API:
    SliderState, BlockState — slider configuration data model.
    RepairEngine — owns the Klein pipeline + primary/donor networks; exposes
        generate_preview() as the single preview entry point (v2 hook).
    save_repaired_lora() — bake slider state into a new .safetensors.
"""

from fizgig.repair_studio.state import BlockState, SliderState
from fizgig.repair_studio.engine import RepairEngine
from fizgig.repair_studio.bake import save_repaired_lora

__all__ = ["BlockState", "SliderState", "RepairEngine", "save_repaired_lora"]
