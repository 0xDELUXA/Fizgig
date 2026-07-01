"""INT8 (W8A8) dynamic quantization for a frozen DiT — experimental inference speedup.

Prototype path for the INT8 experiment. Mirrors modules/fp8.py + modules/nf4.py: keep each
`nn.Linear` (so LoRA targeting by module path is unchanged) and monkey-patch its `forward` to run
an int8 matmul instead of the bf16/fp8 one. Weights are quantized once to int8 with a per-output-
channel scale; activations are quantized dynamically per-token (per row) each forward. The matmul
is `torch._int_mm` (int8 x int8 -> int32), then dequantized back to the activation dtype.

Why int8: on Blackwell (RTX 50-series) the int8 tensor cores are ~1.4-1.8x faster than fp8 at the
matmul level (measured on a 5090), and INT8 cores exist on 30-series too (which have no fast fp8).
This module is the experiment to see whether that kernel win survives the per-matmul quant overhead
across a full forward, and at what quality (naive symmetric W8A8; a rotation/ConvRot pass can be
added later if outliers hurt).

Inference-only — the dynamic activation quant is not differentiable, so this is for previews /
the workbench, not the training base (fp8 / NF4 stay there).
"""
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_INT8_TARGET_DEFAULT = ("blocks.",)


def int8_linear_forward_patch(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Patched forward: per-token dynamic int8 activation quant -> int8 matmul -> dequant.

    Symmetric (no zero point). The frozen weight is pre-quantized to int8 with a per-output-channel
    scale (`_int8_wt` is the (K, N) transposed int8 weight ready for `_int_mm`; `_int8_wscale` is
    (1, N)). Dequant: out[m,n] = (sum_k a_i8[m,k] * w_i8[k,n]) * a_scale[m] * w_scale[n].
    """
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    # Per-token (per-row) symmetric activation quant.
    a_scale = x2d.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0     # (M, 1)
    a_i8 = (x2d / a_scale).round_().clamp_(-127, 127).to(torch.int8)
    try:
        acc = torch._int_mm(a_i8, self._int8_wt)                               # (M, N) int32
    except Exception:
        # Shape-constraint edge case (tiny M, odd alignment) — fall back to a bf16 matmul from the
        # dequantized int8 weight so a preview never crashes on a weird resolution.
        w = (self._int8_wt.to(torch.float32) * self._int8_wscale)             # (K, N) fp32
        out = (x2d.to(torch.float32) @ w).to(x.dtype)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], -1)
    out = acc.to(torch.float32) * a_scale.to(torch.float32) * self._int8_wscale  # (M, N) fp32
    out = out.to(x.dtype)
    if self.bias is not None:
        out = out + self.bias
    return out.reshape(*orig_shape[:-1], -1)


def apply_int8_quantization(
    model: nn.Module,
    target_keys=_INT8_TARGET_DEFAULT,
    exclude_keys=(),
    compute_device: torch.device = torch.device("cuda"),
) -> int:
    """Quantize the target Linears' weights to int8 (per-output-channel scale) in place and patch
    their forward to the int8 path. Reuses nf4's source-weight dequant so an fp8 or bf16 base both
    work. Returns the number of Linears quantized."""
    from fizgig.modules.nf4 import _dequantize_source_weight

    compute_device = torch.device(compute_device)
    count = 0
    for name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) and getattr(module, "weight", None) is not None):
            continue
        if not any(t in name for t in target_keys):
            continue
        if any(e in name for e in exclude_keys):
            continue

        w_bf16 = _dequantize_source_weight(module).to(compute_device).contiguous()   # (N, K)
        w_scale = w_bf16.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0     # (N, 1)
        w_i8 = (w_bf16 / w_scale).round_().clamp_(-127, 127).to(torch.int8)           # (N, K)

        module._int8_wt = w_i8.t().contiguous()                                       # (K, N)
        module._int8_wscale = w_scale.reshape(1, -1).to(torch.float32)                # (1, N)
        module._is_int8 = True
        # Free the original weight (keep the Parameter object so in/out_features + LoRA targeting
        # stay intact); the int8 copy lives in _int8_wt.
        module.weight.data = torch.empty(0, device=compute_device, dtype=torch.bfloat16)
        module.weight.requires_grad_(False)
        module.forward = int8_linear_forward_patch.__get__(module, type(module))

        del w_bf16, w_i8
        count += 1

    if count > 0:
        model._int8_quantized = True
    logger.info(f"INT8 quantization: {count} Linears -> W8A8 (per-channel weight, per-token activation).")
    return count
