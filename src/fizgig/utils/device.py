"""Device utilities for memory management and synchronization."""

import gc
from typing import Optional, Union

import torch


def fp8_scaled_mm_supported(device: Optional[Union[str, torch.device]] = None) -> bool:
    """True if the GPU has fp8 tensor cores usable by torch._scaled_mm.

    Requires compute capability >= 8.9 (Ada / Hopper / Blackwell). Older cards
    (Ampere sm_86 like the 3090, Turing, etc.) lack fp8 silicon and must fall
    back to the dequantize-to-bf16 path — for them this returns False and the
    fast path is never entered, so training/inference behaves exactly as today.
    """
    if not torch.cuda.is_available():
        return False
    if device is not None:
        dev = torch.device(device) if isinstance(device, str) else device
        index = dev.index if dev.type == "cuda" else None
    else:
        index = None
    try:
        major, minor = torch.cuda.get_device_capability(index)
    except Exception:
        return False
    return (major, minor) >= (8, 9)


def clean_memory_on_device(device: Optional[Union[str, torch.device]]):
    """Free cached memory on the specified device."""
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def gpu_svd(W: torch.Tensor) -> tuple:
    """SVD on GPU if available, CPU fallback. Returns (U, S, Vt) on CPU."""
    if torch.cuda.is_available():
        try:
            W_gpu = W.cuda()
            U, S, Vt = torch.linalg.svd(W_gpu, full_matrices=False)
            return U.cpu(), S.cpu(), Vt.cpu()
        except Exception:
            pass
    return torch.linalg.svd(W, full_matrices=False)


def gpu_kron(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Kronecker product on GPU if available, CPU fallback. Returns result on CPU."""
    if torch.cuda.is_available():
        try:
            result = torch.kron(w1.cuda(), w2.cuda()).cpu()
            return result
        except Exception:
            pass
    return torch.kron(w1, w2)


def synchronize_device(device: Optional[Union[str, torch.device]]):
    """Block until all pending operations on the device are complete."""
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
