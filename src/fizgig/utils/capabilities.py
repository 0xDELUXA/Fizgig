"""What this machine can actually do, probed once and cached.

Fizgig used to pick memory settings from VRAM alone, which produced a bad outcome on 16 GB
cards: fp8 doesn't fit, so it fell back to swapping 20 of 28 blocks to CPU every step. Measured
on an RTX 5090 (Krea 2, 36 images @ 0.25 MP, batch 1):

    fp8, no swap    0.85 s/it   20.1 GB   12.5% CPU
    fp8, swap 20    3.09 s/it   12.3 GB   49.9% CPU     <- what 16 GB cards were getting
    NF4, no swap    0.70 s/it   13.8 GB   14.0% CPU

Block swap costs 4.4x the time and 4x the CPU, and NF4 fits the same card outright. So the
choice is a *strategy*, not a swap count — and it needs to know what the hardware supports,
because fp8 matmul is Ada+ while NF4 and int8 go back further.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Capabilities:
    has_cuda: bool = False
    device_name: str = "cpu"
    sm: tuple = (0, 0)
    vram_gb: float = 0.0
    fp8_matmul: bool = False       # torch._scaled_mm on fp8 — Ada (sm 8.9) and newer
    int8_matmul: bool = False      # torch._scaled_mm on int8
    cudnn_attention: bool = False  # PyTorch SDPA cuDNN backend
    flash_attn: bool = False       # the flash_attn package
    bitsandbytes: bool = False     # required for NF4
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        if not self.has_cuda:
            return "no CUDA device"
        flags = [f"sm_{self.sm[0]}{self.sm[1]}"]
        for name, ok in (("fp8", self.fp8_matmul), ("int8", self.int8_matmul),
                         ("cuDNN-attn", self.cudnn_attention), ("flash", self.flash_attn),
                         ("nf4", self.bitsandbytes)):
            flags.append(f"{name} {'yes' if ok else 'no'}")
        return f"{self.device_name}, {self.vram_gb:.0f} GB — " + " · ".join(flags)


def _probe_scaled_mm(dtype) -> bool:
    """Actually run a tiny _scaled_mm rather than trusting a compute-capability table."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        a = torch.zeros((16, 16), dtype=dtype, device="cuda")
        b = torch.zeros((16, 16), dtype=dtype, device="cuda").t()
        one = torch.ones((), dtype=torch.float32, device="cuda")
        torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def detect() -> Capabilities:
    caps = Capabilities()
    try:
        import torch
    except Exception:
        caps.notes.append("torch not importable")
        return caps

    if not torch.cuda.is_available():
        caps.notes.append("CUDA unavailable")
        return caps

    caps.has_cuda = True
    props = torch.cuda.get_device_properties(0)
    caps.device_name = props.name
    caps.sm = torch.cuda.get_device_capability(0)
    caps.vram_gb = props.total_memory / (1024 ** 3)

    caps.fp8_matmul = _probe_scaled_mm(torch.float8_e4m3fn)
    caps.int8_matmul = _probe_scaled_mm(torch.int8)

    try:    # cuDNN SDPA backend: present from PyTorch 2.5-ish, Ampere and newer
        from torch.backends.cuda import can_use_cudnn_attention  # noqa: F401
        caps.cudnn_attention = True
    except Exception:
        caps.cudnn_attention = hasattr(__import__("torch").backends.cuda, "cudnn_sdp_enabled")

    try:
        import flash_attn  # noqa: F401
        caps.flash_attn = True
    except Exception:
        pass

    try:
        import bitsandbytes  # noqa: F401
        caps.bitsandbytes = True
    except Exception:
        caps.notes.append("bitsandbytes missing — NF4 unavailable")

    return caps


# TRAINING-ONLY footprints at 0.25 MP, batch 1: the measured peaks (20.1 / 13.8 GB) were
# whole-GPU readings on a desktop already holding ~2.4 GB, so they overstate what training
# needs. Headroom then covers the user's own desktop plus allocator slack.
#
# Note the comparison is against TOTAL reported VRAM, and a "16 GB" card reports ~15.9 GiB —
# thresholds have to clear that, not 16.0, or 16 GB cards fall through to swapping. That is
# exactly the off-by-a-fraction that made them 4.4x slower before this.
_FP8_PEAK_GB = 17.7
_NF4_PEAK_GB = 11.4
_HEADROOM_GB = 3.0


@dataclass
class MemoryStrategy:
    quant_4bit: bool
    blocks_to_swap: int
    reason: str


def recommend_krea2_strategy(vram_gb: Optional[float] = None,
                             caps: Optional[Capabilities] = None) -> MemoryStrategy:
    """Pick quantisation + swap for Krea 2 training on this machine.

    Preference order: NF4 with no swap > fp8 with no swap > fp8 with swap. Swapping is last
    because it is 4.4x slower and 4x the CPU load; NF4 leads because it measured *faster* than
    fp8 as well as smaller (its dequant is a fused bitsandbytes kernel, whereas the fp8 path
    materialises a bf16 copy of each weight per forward).
    """
    caps = caps or detect()
    vram = vram_gb if vram_gb is not None else caps.vram_gb

    if not caps.has_cuda:
        return MemoryStrategy(False, 0, "no CUDA device — settings left alone")

    if caps.bitsandbytes and vram >= _NF4_PEAK_GB + _HEADROOM_GB:
        return MemoryStrategy(
            True, 0,
            f"NF4 4-bit, no block swap (~{_NF4_PEAK_GB:.0f} GB of {vram:.0f} GB) — "
            "fastest measured and leaves the most headroom")

    if vram >= _FP8_PEAK_GB + _HEADROOM_GB:
        return MemoryStrategy(
            False, 0, f"fp8, no block swap (~{_FP8_PEAK_GB:.0f} GB of {vram:.0f} GB)")

    if not caps.bitsandbytes:
        swap = 12 if vram >= 22 else (20 if vram >= 15 else 26)
        return MemoryStrategy(
            False, swap,
            f"fp8 with {swap} blocks swapped — bitsandbytes is missing, so NF4 (which would "
            "avoid swapping entirely and run ~4x faster) is unavailable. Install it.")

    # Below NF4's own footprint: swap on top of 4-bit is the only way to fit.
    swap = 12 if vram >= 11 else 20
    return MemoryStrategy(
        True, swap,
        f"NF4 4-bit with {swap} blocks swapped — {vram:.0f} GB is below what Krea 2 needs "
        "resident even at 4-bit, so some swapping is unavoidable")
