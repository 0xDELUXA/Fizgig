"""Which SDPA backend to use, and when — one decision, shared by every attention site.

cuDNN's SDPA kernel is much faster than PyTorch's default choice for these shapes — as long as
the shape STAYS THE SAME. It plans a kernel per shape and re-plans whenever the shape changes.
Measured on Krea 2's attention shape (48q/12kv heads, head_dim 128, bf16, RTX 5090, fwd+bwd):

    fixed shape (one sequence length, repeated)     default 1.904 ms   cuDNN  0.656 ms
    varying shapes (576/900/1024/1156/1444 tokens)  default 2.248 ms   cuDNN 26.353 ms

Bucketed training cycles through a different token count per aspect ratio, so it pays that
re-planning cost on nearly every step:

    1024 preview (one shape, 8 steps)   5.15 s     -> 4.40 s     with cuDNN
    training step (bucketed)            0.704 s/it -> 1.57 s/it  with cuDNN

NOT a backward-pass problem — cuDNN's backward is fine, and is 2.7-3.1x faster than the default
at a fixed shape with or without an attention mask.

cudnn.benchmark matters enormously and in the opposite direction to intuition. PyTorch defaults
it to False, which puts cuDNN on a heuristic path that collapses under shape churn; letting it
autotune costs one benchmark per shape and then caches:

    varying shapes, cudnn.benchmark=False .... 51.573 ms
    varying shapes, cudnn.benchmark=True .....  0.757 ms   (68x better, and 2.9x the default)

It is therefore enabled whenever cuDNN is selected. In isolation that ought to make cuDNN win for
training too — but end to end it does NOT: 1.57 -> 1.28 s/it with autotuning, against 0.704 for
the default backend. Three cuDNN variants (plain / bucket-grouped / autotuned) all lose, so
something in the real attention call (variable-shaped masks, the split-attn path) costs more than
the isolated kernel benchmark shows. Inference-only stands, on measurement rather than theory.

`torch.is_grad_enabled()` is the switch because it tracks that distinction exactly: False under
no_grad (previews, Royale, Repair Studio, Explorer, sampling — one resolution held for a whole
render) and True in a training step, including inside a gradient-checkpoint recompute. If a
fixed-shape trainer ever appears, revisit the test rather than the conclusion.

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
            torch.backends.cudnn.benchmark = True    # see the module docstring — 68x
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
