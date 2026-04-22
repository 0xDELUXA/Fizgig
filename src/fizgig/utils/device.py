"""Device utilities for memory management and synchronization."""

import gc
from typing import Optional, Union

import torch


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
