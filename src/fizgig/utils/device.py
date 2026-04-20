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
