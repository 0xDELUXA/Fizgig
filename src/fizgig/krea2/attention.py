# Unified attention function supporting various implementations
# Adapted from musubi-tuner (https://github.com/kohya-ss/musubi-tuner), Apache-2.0.
# Modified for Fizgig. See THIRD_PARTY_NOTICES.md.

from dataclasses import dataclass
import logging
import torch
from typing import Optional, Union

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from flash_attn.flash_attn_interface import flash_attn_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None
    flash_attn_func = None

try:
    from sageattention import sageattn_varlen, sageattn
except ImportError:
    sageattn_varlen = None
    sageattn = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None


@dataclass
class AttentionParams:
    attn_mode: Optional[str] = None
    split_attn: bool = False
    img_len: Optional[int] = None
    attention_mask: Optional[torch.Tensor] = None
    seqlens: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    max_seqlen: Optional[int] = None

    @staticmethod
    def create_attention_params(attn_mode: Optional[str], split_attn: bool) -> "AttentionParams":
        return AttentionParams(attn_mode, split_attn)

    @staticmethod
    def create_attention_params_from_mask(
        attn_mode: Optional[str], split_attn: bool, img_len: Optional[int], attention_mask: Optional[torch.Tensor]
    ) -> "AttentionParams":
        if attention_mask is None:
            # No attention mask provided: assume all tokens are valid
            return AttentionParams(attn_mode, split_attn, None, None, None, None, None)
        else:
            # Note: attention_mask is only for text tokens, not including image tokens
            seqlens = attention_mask.sum(dim=1).to(torch.int32) + img_len  # [B]
            max_seqlen = attention_mask.shape[1] + img_len

            if split_attn:
                # cu_seqlens is not needed for split attention
                return AttentionParams(attn_mode, split_attn, img_len, attention_mask, seqlens, None, max_seqlen)

            # Convert attention mask to cumulative sequence lengths for flash attention
            batch_size = attention_mask.shape[0]
            cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device=attention_mask.device)
            for i in range(batch_size):
                cu_seqlens[2 * i + 1] = i * max_seqlen + seqlens[i]  # end of valid tokens for query
                cu_seqlens[2 * i + 2] = (i + 1) * max_seqlen  # end of all tokens for query

            # Expand attention mask to include image tokens
            attention_mask = torch.nn.functional.pad(attention_mask, (img_len, 0), value=1)  # [B, img_len + L]

            # attention bias for xformers
            if attn_mode == "xformers":
                seqlens_list = seqlens.cpu().tolist()
                attention_mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                    seqlens_list, seqlens_list, device=attention_mask.device
                )
            elif attn_mode == "torch":
                attention_mask = attention_mask[:, None, None, :].to(torch.bool)  # [B, 1, 1, img_len + L]

            return AttentionParams(attn_mode, split_attn, img_len, attention_mask, seqlens, cu_seqlens, max_seqlen)


# cuDNN's SDPA kernel is much faster than PyTorch's default choice for Krea 2's shapes — but
# only in the FORWARD direction. Measured (48q/12kv heads, head_dim 128, bf16, RTX 5090):
#
#     forward only:  seq 1024  default 0.330 ms   cuDNN 0.128 ms   (2.6x)
#                    seq 4096  default 4.664 ms   cuDNN 1.841 ms   (2.5x)
#
# End-to-end, however, the two diverge completely:
#
#     1024 preview (no grad) .......... 5.15 s -> 4.40 s   with cuDNN
#     training step (fwd+bwd) ......... 0.70  -> 1.57 s/it with cuDNN   <- 2.2x SLOWER
#
# cuDNN's backward is the problem, and a forward-only microbenchmark hides it entirely. So the
# backend is selected on whether gradients are being recorded, not globally.
#
# Numerically equivalent to bf16 tolerance (rel-err ~5e-4, masked or not). Probed once; if the
# kernel is unavailable this is a no-op and PyTorch chooses as before. FIZGIG_SDPA_BACKEND=
# default|cudnn overrides.
logger = logging.getLogger(__name__)

_SDPA_CTX = None


def _sdpa_backend_ctx():
    """cuDNN SDPA for inference; PyTorch's default when gradients are being recorded."""
    import contextlib, os
    choice = os.environ.get("FIZGIG_SDPA_BACKEND", "auto").lower()
    if choice in ("default", "off", "none"):
        return contextlib.nullcontext()
    # Training: cuDNN's backward is 2.2x slower end-to-end (see above). is_grad_enabled is the
    # right test — it is False under torch.no_grad (previews, the workbench, sampling) and True
    # in a training step, including inside a gradient-checkpoint recompute.
    if torch.is_grad_enabled() and choice != "cudnn":
        return contextlib.nullcontext()

    global _SDPA_CTX
    if _SDPA_CTX is None:
        _SDPA_CTX = contextlib.nullcontext
        try:
            import torch.nn.functional as _F
            from torch.nn.attention import sdpa_kernel, SDPBackend
            _q = torch.zeros(1, 8, 64, 64, device="cuda", dtype=torch.bfloat16)
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                _F.scaled_dot_product_attention(_q, _q, _q)
            _SDPA_CTX = lambda: sdpa_kernel(SDPBackend.CUDNN_ATTENTION)
            logger.info("[attention] using the cuDNN SDPA backend (~2.5x faster than the default here)")
        except Exception as e:
            logger.info("[attention] cuDNN SDPA backend unavailable (%s) — using PyTorch's default choice",
                        type(e).__name__)
    return _SDPA_CTX()


def attention(
    qkv_or_q: Union[torch.Tensor, list],
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    attn_params: Optional[AttentionParams] = None,
    drop_rate: float = 0.0,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention with variable sequence lengths.

    Handles batches with different sequence lengths by splitting and
    processing each sequence individually.

    Args:
        qkv_or_q: Query tensor [B, L, H, D]. or list of such tensors.
        k: Key tensor [B, L, H, D].
        v: Value tensor [B, L, H, D].
        attn_param: Attention parameters including mask and sequence lengths.
        drop_rate: Attention dropout rate.

    Returns:
        Attention output tensor [B, L, H*D].
    """
    if isinstance(qkv_or_q, list):
        q, k, v = qkv_or_q
        q: torch.Tensor = q
        qkv_or_q.clear()
        del qkv_or_q
    else:
        q: torch.Tensor = qkv_or_q
        del qkv_or_q
        assert k is not None and v is not None, "k and v must be provided if qkv_or_q is a tensor"
    if attn_params is None:
        attn_params = AttentionParams.create_attention_params("torch", False)

    # GQA: q may carry more heads than k/v (e.g. Krea 2 = 48 query / 12 kv heads). Expanding
    # k/v stays the right call even on cuDNN — measured 0.128 ms (expand) vs 0.138 ms
    # (enable_gqa) at seq 1024, and enable_gqa on the DEFAULT backend is 1.802 ms. flash and
    # sageattn group heads natively inside the kernel (verified), so they ignore this. For the
    # torch (SDPA) path we expand k/v to q's head count below instead of passing enable_gqa=True,
    # because enable_gqa forces SDPA onto the slow math kernel (~7x slower than the fused kernels
    # at K2 scale); the repeat is numerically identical. Revert to enable_gqa=True once PyTorch's
    # fused (flash / mem-efficient) kernels gain native GQA support. (q/k/v here are [B, L, H, D].)
    enable_gqa = q.shape[-2] != k.shape[-2]

    # If split attn is False, attention mask is provided and all sequence lengths are same, we can trim the sequence
    seqlen_trimmed = False
    # Trim if all seqlens are the same, for attention modes other than flash or sageattn (which can handle masks efficiently)
    if (
        not attn_params.split_attn
        and attn_params.attention_mask is not None
        and attn_params.seqlens is not None
        and (attn_params.attn_mode != "flash" and attn_params.attn_mode != "sageattn")
    ):
        if torch.all(attn_params.seqlens == attn_params.seqlens[0]):
            seqlen = attn_params.seqlens[0].item()
            q = q[:, :seqlen]
            k = k[:, :seqlen]
            v = v[:, :seqlen]
            max_seqlen = attn_params.max_seqlen
            attn_params = AttentionParams.create_attention_params(attn_params.attn_mode, False)  # do not in-place modify
            attn_params.max_seqlen = max_seqlen  # keep max_seqlen for padding
            seqlen_trimmed = True

    # Determine tensor layout based on attention implementation
    if attn_params.attn_mode == "torch" or (
        attn_params.attn_mode == "sageattn" and (attn_params.split_attn or attn_params.cu_seqlens is None)
    ):
        transpose_fn = lambda x: x.transpose(1, 2)  # [B, H, L, D] for SDPA and sageattn with fixed length
        # pad on sequence length dimension
        pad_fn = lambda x, pad_to: torch.nn.functional.pad(x, (0, 0, 0, pad_to - x.shape[-2]), value=0)
    else:
        transpose_fn = lambda x: x  # [B, L, H, D] for other implementations
        # pad on sequence length dimension
        pad_fn = lambda x, pad_to: torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_to - x.shape[-3]), value=0)

    # Process each batch element with its valid sequence lengths
    if attn_params.split_attn:
        if attn_params.seqlens is None:
            # If no seqlens provided, assume all tokens are valid
            attn_params = AttentionParams.create_attention_params(attn_params.attn_mode, True)  # do not in-place modify
            attn_params.seqlens = torch.tensor([q.shape[1]] * q.shape[0], device=q.device)
            attn_params.max_seqlen = q.shape[1]
        q = [transpose_fn(q[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(q))]
        k = [transpose_fn(k[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(k))]
        v = [transpose_fn(v[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(v))]
    else:
        q = transpose_fn(q)
        k = transpose_fn(k)
        v = transpose_fn(v)

    if attn_params.attn_mode == "torch":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                qi, ki, vi = q[i], k[i], v[i]
                if enable_gqa:  # expand k/v heads to avoid SDPA's slow enable_gqa math path
                    g = qi.shape[1] // ki.shape[1]  # [B, H, L, D] -> heads at dim 1
                    ki = ki.repeat_interleave(g, dim=1)
                    vi = vi.repeat_interleave(g, dim=1)
                with _sdpa_backend_ctx():
                    x_i = torch.nn.functional.scaled_dot_product_attention(qi, ki, vi, dropout_p=drop_rate)
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, H, L, D
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None

        else:
            if enable_gqa:  # expand k/v heads to avoid SDPA's slow enable_gqa math path
                g = q.shape[1] // k.shape[1]  # [B, H, L, D] -> heads at dim 1
                k = k.repeat_interleave(g, dim=1)
                v = v.repeat_interleave(g, dim=1)
            with _sdpa_backend_ctx():
                x = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_params.attention_mask, dropout_p=drop_rate)
            q, k, v = None, None, None

    elif attn_params.attn_mode == "xformers":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                qi, ki, vi = q[i], k[i], v[i]
                if enable_gqa:  # expand k/v heads to q's count; xformers 4D has no native GQA ([B, L, H, D] -> dim 2)
                    g = qi.shape[2] // ki.shape[2]
                    ki = ki.repeat_interleave(g, dim=2)
                    vi = vi.repeat_interleave(g, dim=2)
                x_i = xops.memory_efficient_attention(qi, ki, vi, p=drop_rate)
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, L, H, D
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None

        else:
            if enable_gqa:  # expand k/v heads to q's count; xformers 4D has no native GQA ([B, L, H, D] -> dim 2)
                g = q.shape[2] // k.shape[2]
                k = k.repeat_interleave(g, dim=2)
                v = v.repeat_interleave(g, dim=2)
            x = xops.memory_efficient_attention(q, k, v, attn_bias=attn_params.attention_mask, p=drop_rate)
            q, k, v = None, None, None

    elif attn_params.attn_mode == "sageattn":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                # HND seems to cause an error
                x_i = sageattn(q[i], k[i], v[i])  # B, H, L, D. No dropout support
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, H, L, D
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None
        elif attn_params.cu_seqlens is None:  # all tokens are valid
            x = sageattn(q, k, v)  # B, L, H, D. No dropout support
            q, k, v = None, None, None
        else:
            # Reshape to [(bxs), a, d]
            batch_size, seqlen = q.shape[0], q.shape[1]
            # reshape (not view): callers may pass non-contiguous q/k/v (e.g. Krea 2 transposes
            # [B,H,L,D]->[B,L,H,D] after RoPE); reshape copies only when needed, view-equivalent otherwise.
            q = q.reshape(q.shape[0] * q.shape[1], *q.shape[2:])  # [B*L, H, D]
            k = k.reshape(k.shape[0] * k.shape[1], *k.shape[2:])  # [B*L, H, D]
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])  # [B*L, H, D]

            # Assume cu_seqlens_q == cu_seqlens_kv and max_seqlen_q == max_seqlen_kv. No dropout support
            x = sageattn_varlen(
                q, k, v, attn_params.cu_seqlens, attn_params.cu_seqlens, attn_params.max_seqlen, attn_params.max_seqlen
            )
            q, k, v = None, None, None

            # Reshape x with shape [(bxs), a, d] to [b, s, a, d]
            x = x.view(batch_size, seqlen, x.shape[-2], x.shape[-1])  # B, L, H, D

    elif attn_params.attn_mode == "flash":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                # HND seems to cause an error
                x_i = flash_attn_func(q[i], k[i], v[i], drop_rate)  # B, L, H, D
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, L, H, D
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None
        elif attn_params.cu_seqlens is None:  # all tokens are valid
            x = flash_attn_func(q, k, v, drop_rate)  # B, L, H, D
            q, k, v = None, None, None
        else:
            # Reshape to [(bxs), a, d]
            batch_size, seqlen = q.shape[0], q.shape[1]
            # reshape (not view): callers may pass non-contiguous q/k/v (e.g. Krea 2 transposes
            # [B,H,L,D]->[B,L,H,D] after RoPE); reshape copies only when needed, view-equivalent otherwise.
            q = q.reshape(q.shape[0] * q.shape[1], *q.shape[2:])  # [B*L, H, D]
            k = k.reshape(k.shape[0] * k.shape[1], *k.shape[2:])  # [B*L, H, D]
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])  # [B*L, H, D]

            # Assume cu_seqlens_q == cu_seqlens_kv and max_seqlen_q == max_seqlen_kv
            x = flash_attn_varlen_func(
                q, k, v, attn_params.cu_seqlens, attn_params.cu_seqlens, attn_params.max_seqlen, attn_params.max_seqlen, drop_rate
            )
            q, k, v = None, None, None

            # Reshape x with shape [(bxs), a, d] to [b, s, a, d]
            x = x.view(batch_size, seqlen, x.shape[-2], x.shape[-1])  # B, L, H, D

    else:
        # Currently only PyTorch SDPA and xformers are implemented
        raise ValueError(f"Unsupported attention mode: {attn_params.attn_mode}")

    x = transpose_fn(x)  # [B, L, H, D]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, L, H*D]

    if seqlen_trimmed:
        x = torch.nn.functional.pad(x, (0, 0, 0, attn_params.max_seqlen - x.shape[1]), value=0)  # pad back to max_seqlen

    return x
