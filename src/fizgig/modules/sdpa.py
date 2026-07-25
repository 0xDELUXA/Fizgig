"""Which SDPA backend to use, and when — one decision, shared by every attention site.

cuDNN's SDPA kernel is faster than PyTorch's default choice for these shapes — but it builds a
plan the first time it sees each shape, and that build is expensive. Measured on the REAL
INT8 Krea 2 model, real cached batches, fwd+bwd+optimizer (RTX 5090):

                      first sight    steady state    one-time, 36 shapes
    default backend     582.2 ms       572.2 ms            0.4 s
    cuDNN              1829.9 ms       536.6 ms           46.6 s

So cuDNN is ~6% FASTER per step once warm, and costs ~1.3 s per distinct shape to get there.
Which one wins is purely a question of how many steps amortize that:

    saves 37 ms/step, costs 46.6 s once  ->  breaks even near 1260 steps

Inference is the clear case FOR it: one resolution held for a whole render, so the plan is built
once and reused for every step of every image.

    Krea 2 1024 preview   5.15 s -> 4.40 s
    Klein  1024 preview   3.01 s -> 2.74 s

Training is the ambiguous case. On a 36-image set that break-even lands around 35 epochs; shorter
runs lose, longer runs win by up to ~6%. The default backend is chosen for training because it is
the one that cannot lose, not because cuDNN is bad — FIZGIG_SDPA_BACKEND=cudnn is there for anyone
doing long runs. Note the cost scales with the number of DISTINCT shapes, which the attention trim
inflates (each caption's token count becomes part of the sequence length — 30 shapes from 36
images); FIZGIG_ATTN_TRIM=0 trades compute for far fewer shapes.

Two earlier conclusions here were WRONG and are corrected above: that cuDNN is ~2x slower for
training (it was a short-benchmark artifact — 46.6 s of plan building charged against 108 steps),
and that cudnn.benchmark is worth 68x (it makes no measurable difference at all: 66.0 s of plan
building with it off vs 65.9 s on, 535.1 vs 536.6 ms/step steady).

`torch.is_grad_enabled()` is the switch because it tracks that distinction exactly: False under
no_grad (previews, Royale, Repair Studio, Explorer, sampling — one resolution held for a whole
render) and True in a training step, including inside a gradient-checkpoint recompute.

Numerically equivalent to bf16 tolerance (rel-err ~5e-4, masked or not). Probed once; if the
kernel is unavailable this is a no-op. FIZGIG_SDPA_BACKEND=default|cudnn overrides.

Used by the DiT attention paths (Krea 2 and Klein, head_dim 128). The VAE attention blocks are
deliberately left alone: they run single-head with head_dim = channels, which cuDNN cannot take
at all, so wrapping them would buy nothing.
"""

from __future__ import annotations

import contextlib
import logging
import os

import torch

logger = logging.getLogger(__name__)

_SDPA_CTX = None


def sdpa_backend_ctx():
    """cuDNN SDPA for inference; PyTorch's default when gradients are being recorded."""
    choice = os.environ.get("FIZGIG_SDPA_BACKEND", "auto").lower()
    if choice in ("default", "off", "none"):
        return contextlib.nullcontext()
    # Training means bucketing means shape churn, which is what cuDNN handles badly (see above).
    if torch.is_grad_enabled() and choice != "cudnn":
        return contextlib.nullcontext()

    global _SDPA_CTX
    if _SDPA_CTX is None:
        _SDPA_CTX = contextlib.nullcontext
        try:
            import torch.nn.functional as _F
            from torch.nn.attention import sdpa_kernel, SDPBackend
            # cudnn.benchmark is NOT set here. It used to be, justified by a 68x figure that
            # later failed to reproduce; measured on the real model it changes nothing at all —
            # neither the per-shape plan cost (66.0 s off vs 65.9 s on, over 36 shapes) nor the
            # steady state (535.1 vs 536.6 ms/step). Setting a global torch flag with no
            # measurable effect is worse than not setting it. FIZGIG_CUDNN_BENCHMARK=1 opts in.
            if os.environ.get("FIZGIG_CUDNN_BENCHMARK", "0") != "0":
                torch.backends.cudnn.benchmark = True
            # A PREFERENCE, not a demand. Forcing the single backend raises "No available
            # kernel" on anything cuDNN can't take — head_dim > 128, which is exactly the
            # VAE's single-head attention. The priority list keeps the win where cuDNN is
            # eligible (0.128 ms vs 0.133 forced, against 0.331 for PyTorch's own choice at
            # Krea 2's shape) and quietly falls back everywhere else.
            _backends = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                         SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            _q = torch.zeros(1, 8, 64, 64, device="cuda", dtype=torch.bfloat16)
            with sdpa_kernel(_backends, set_priority=True):
                _F.scaled_dot_product_attention(_q, _q, _q)
            _SDPA_CTX = lambda: sdpa_kernel(_backends, set_priority=True)
            logger.info("[attention] using the cuDNN SDPA backend (~2.5x faster than the default here)")
        except Exception as e:
            logger.info("[attention] cuDNN SDPA backend unavailable (%s) — using PyTorch's default choice",
                        type(e).__name__)
    return _SDPA_CTX()
