"""Token packing and position ID utilities for Klein DiT.

Handles the conversion between spatial (C,H,W) and sequence (HW,C) formats,
with 4D position IDs (time, height, width, layer) for RoPE.
"""

from typing import Optional

import torch
from einops import rearrange
from torch import Tensor


def compress_time(t_ids: Tensor) -> Tensor:
    """Remap time IDs to contiguous range starting from 0."""
    assert t_ids.ndim == 1
    t_ids_max = torch.max(t_ids)
    t_remap = torch.zeros((t_ids_max + 1,), device=t_ids.device, dtype=t_ids.dtype)
    t_unique_sorted_ids = torch.unique(t_ids, sorted=True)
    t_remap[t_unique_sorted_ids] = torch.arange(len(t_unique_sorted_ids), device=t_ids.device, dtype=t_ids.dtype)
    return t_remap[t_ids]


def prc_img(x: Tensor, t_coord: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    """Pack image latents from spatial to sequence format with position IDs.

    Args:
        x: Image latents (C,H,W) or (B,C,H,W).
        t_coord: Optional time coordinate tensor.

    Returns:
        Tuple of (packed_tokens, position_ids):
            packed_tokens: (HW,C) or (B,HW,C)
            position_ids: (HW,4) or (B,HW,4) with dims [t,h,w,l]
    """
    h = x.shape[-2]
    w = x.shape[-1]
    coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(h),
        "w": torch.arange(w),
        "l": torch.arange(1),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    x = rearrange(x, "c h w -> (h w) c") if x.ndim == 3 else rearrange(x, "b c h w -> b (h w) c")
    if x.ndim == 3:  # after rearrange: batched
        x_ids = x_ids.unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, HW, 4)
    return x, x_ids.to(x.device)


def prc_txt(x: Tensor, t_coord: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    """Prepare text embeddings with position IDs.

    Args:
        x: Text embeddings (B,L,D) or (L,D).
        t_coord: Optional time coordinate tensor.

    Returns:
        Tuple of (x, position_ids) where position_ids has dims [t,h,w,l].
    """
    _l = x.shape[-2]
    coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(1),
        "w": torch.arange(1),
        "l": torch.arange(_l),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    if x.ndim == 3:
        x_ids = x_ids.unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, L, 4)
    return x, x_ids.to(x.device)


def _listed_wrapper(fn):
    """Wrap a function to accept lists of tensors."""
    def listed_fn(x: list[Tensor], t_coord: Optional[list[Tensor]] = None) -> tuple[list[Tensor], list[Tensor]]:
        results = []
        for i in range(len(x)):
            results.append(fn(x[i], t_coord[i] if t_coord is not None else None))
        xs, x_ids = zip(*results)
        return list(xs), list(x_ids)
    return listed_fn


listed_prc_img = _listed_wrapper(prc_img)


def scatter_ids(x: Tensor, x_ids: Tensor) -> list[Tensor]:
    """Unpack sequence tokens back to spatial layout using position IDs.

    Args:
        x: Packed tokens (B, seq_len, C).
        x_ids: Position IDs (B, seq_len, 4).

    Returns:
        List of spatial tensors (1, C, T, H, W) per batch element.
    """
    x_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        t_ids_cmpr = compress_time(t_ids)

        t = torch.max(t_ids_cmpr) + 1
        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = t_ids_cmpr * w * h + h_ids * w + w_ids

        out = torch.zeros((t * h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

        x_list.append(rearrange(out, "(t h w) c -> 1 c t h w", t=t, h=h, w=w))
    return x_list


def pack_control_latent(control_latent: Optional[list[Tensor]]) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Pack control/reference image latents into token sequence.

    Args:
        control_latent: List of control latent tensors, or None.

    Returns:
        Tuple of (ref_tokens, ref_ids) or (None, None).
    """
    if control_latent is None:
        return None, None

    scale = 10
    ndim = control_latent[0].ndim  # 3 or 4

    t_off = [scale + scale * t for t in torch.arange(0, len(control_latent))]
    t_off = [t.view(-1) for t in t_off]

    ref_tokens, ref_ids = listed_prc_img(control_latent, t_coord=t_off)

    cat_dimension = 0 if ndim == 3 else 1
    ref_tokens = torch.cat(ref_tokens, dim=cat_dimension)
    ref_ids = torch.cat(ref_ids, dim=cat_dimension)

    if ndim == 3:
        ref_tokens = ref_tokens.unsqueeze(0)
        ref_ids = ref_ids.unsqueeze(0)

    return ref_tokens, ref_ids
