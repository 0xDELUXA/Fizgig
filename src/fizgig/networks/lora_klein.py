"""Klein-specific LoRA network configuration.

Defines the target modules and default exclusion patterns for applying
LoRA to Klein 9B's DoubleStreamBlock and SingleStreamBlock layers.
"""

import ast
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import logging

import fizgig.networks.lora as lora

logger = logging.getLogger(__name__)

# Klein 9B LoRA targets: these block classes contain the Linear layers we inject LoRA into
KLEIN_TARGET_REPLACE_MODULES = ["DoubleStreamBlock", "SingleStreamBlock"]


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> lora.LoRANetwork:
    """Create a Klein-specific LoRA network with default exclusion patterns.

    Default excludes: modulation linear layers and all norm layers.
    These are not useful LoRA targets and excluding them matches the
    ComfyUI-compatible key format.
    """
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = [r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*"]
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    # Also exclude norm layers
    exclude_patterns.append(r".*(norm).*")
    kwargs["exclude_patterns"] = exclude_patterns

    return lora.create_network(
        KLEIN_TARGET_REPLACE_MODULES,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    """Create a Klein LoRA network from existing weight files."""
    return lora.create_network_from_weights(
        KLEIN_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
