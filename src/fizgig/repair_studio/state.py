"""Slider state data model for the Repair Studio.

A SliderState is the *complete* configuration of the studio at a point in time:
per-block primary/donor enables and strengths, plus the preview parameters
(seed, prompt, resolution). It serializes to/from JSON for preset storage.

The state is the SOLE input to RepairEngine.generate_preview(). Keeping it as
a single dataclass lets v2 diff successive states cheaply for cache invalidation.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


# Klein 9B block layout: 8 double + 24 single = 32 sliders.
DOUBLE_BLOCK_COUNT = 8
SINGLE_BLOCK_COUNT = 24


def all_block_ids() -> List[str]:
    ids = [f"double_{i}" for i in range(DOUBLE_BLOCK_COUNT)]
    ids += [f"single_{i}" for i in range(SINGLE_BLOCK_COUNT)]
    return ids


def block_regex(block_id: str) -> str:
    """Return a regex matching a LoRA module's lora_name for this block.

    LoRA module names follow `lora_unet_<double|single>_blocks_<N>_<linear>`.
    set_module_multiplier_by_pattern uses regex.search on lora_name, so anchoring
    on the underscore after the index is enough to avoid `single_1_` matching
    `single_10_*`.
    """
    kind, idx = block_id.split("_")
    return rf"{kind}_blocks_{idx}_"


@dataclass
class BlockState:
    primary_enabled: bool = True
    primary_strength: float = 1.0
    donor_enabled: bool = False
    donor_strength: float = 1.0


@dataclass
class SliderState:
    blocks: Dict[str, BlockState] = field(default_factory=dict)
    seed: int = 42
    prompt: str = ""
    preview_width: int = 512
    preview_height: int = 512

    @classmethod
    def default_klein9b(cls) -> "SliderState":
        return cls(blocks={bid: BlockState() for bid in all_block_ids()})

    def to_json(self) -> Dict[str, Any]:
        return {
            "blocks": {bid: asdict(bs) for bid, bs in self.blocks.items()},
            "seed": self.seed,
            "prompt": self.prompt,
            "preview_width": self.preview_width,
            "preview_height": self.preview_height,
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "SliderState":
        blocks = {bid: BlockState(**bs) for bid, bs in d.get("blocks", {}).items()}
        if not blocks:
            blocks = {bid: BlockState() for bid in all_block_ids()}
        return cls(
            blocks=blocks,
            seed=int(d.get("seed", 42)),
            prompt=str(d.get("prompt", "")),
            preview_width=int(d.get("preview_width", 512)),
            preview_height=int(d.get("preview_height", 512)),
        )

    def copy(self) -> "SliderState":
        """Fast deep copy without JSON serialization."""
        return SliderState(
            blocks={bid: BlockState(bs.primary_enabled, bs.primary_strength,
                                    bs.donor_enabled, bs.donor_strength)
                    for bid, bs in self.blocks.items()},
            seed=self.seed, prompt=self.prompt,
            preview_width=self.preview_width, preview_height=self.preview_height,
        )

    def mutate(self, active_blocks: set, num_mutations: int = 3,
               intensity: float = 0.5) -> "SliderState":
        """Return a new SliderState with random perturbations to num_mutations blocks.

        active_blocks: set of block_ids the LoRA touches (skip others).
        intensity: 0.0 = tiny nudges (±0.1), 1.0 = bold moves (±2.0).
        """
        import random
        new = self.copy()
        candidates = [bid for bid in new.blocks if bid in active_blocks]
        if not candidates:
            return new
        chosen = random.sample(candidates, min(num_mutations, len(candidates)))
        for bid in chosen:
            bs = new.blocks[bid]
            # 0.0 → ±0.1, 0.5 → ±1.05, 1.0 → ±2.0 (linear ramp)
            magnitude = 0.1 + intensity * 1.9
            delta = random.uniform(-magnitude, magnitude)
            bs.primary_strength = max(-3.0, min(3.0, bs.primary_strength + delta))
            # Chance of toggling enable scales with intensity (0% at 0, 25% at max)
            if random.random() < 0.25 * intensity:
                bs.primary_enabled = not bs.primary_enabled
        return new

    def diff_blocks(self, other: "SliderState") -> List[str]:
        """Block ids whose primary/donor enable or strength differs from `other`.
        Used by the UI to feed RepairEngine.mark_blocks_changed (v2 hook)."""
        changed = []
        for bid, bs in self.blocks.items():
            ob = other.blocks.get(bid)
            if ob is None or bs != ob:
                changed.append(bid)
        return changed
