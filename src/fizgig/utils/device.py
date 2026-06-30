"""Device utilities for memory management and synchronization."""

import gc
import logging
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


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
    """SVD on GPU if available, CPU fallback. Returns (U, S, Vt) on CPU.

    A CUDA failure (usually OOM when the GPU is busy) falls back to CPU — but it's logged
    at WARNING so a slow, CPU-bound run is diagnosable instead of a silent mystery."""
    if torch.cuda.is_available():
        try:
            W_gpu = W.cuda()
            U, S, Vt = torch.linalg.svd(W_gpu, full_matrices=False)
            return U.cpu(), S.cpu(), Vt.cpu()
        except Exception as e:
            logger.warning("gpu_svd: CUDA SVD failed (%s: %s) for shape %s — falling back to CPU "
                           "(much slower). Free GPU memory to keep SVD on the GPU.",
                           type(e).__name__, e, tuple(W.shape))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return torch.linalg.svd(W, full_matrices=False)


def gpu_kron(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Kronecker product on GPU if available, CPU fallback. Returns result on CPU.

    Logs at WARNING on CUDA failure so a CPU fallback (usually OOM) is visible, not silent."""
    if torch.cuda.is_available():
        try:
            result = torch.kron(w1.cuda(), w2.cuda()).cpu()
            return result
        except Exception as e:
            logger.warning("gpu_kron: CUDA kron failed (%s: %s) for shapes %s x %s — falling back "
                           "to CPU. Free GPU memory to keep this on the GPU.",
                           type(e).__name__, e, tuple(w1.shape), tuple(w2.shape))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
