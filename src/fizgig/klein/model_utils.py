"""Model loading utilities for Klein 9B.

Provides functions to load the KleinDiT, AutoEncoder (VAE), and Qwen3 text embedder,
with support for FP8 optimization and LoRA weight merging during loading.
"""

import argparse
import math
import os
import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from tqdm import tqdm

from fizgig.training.metadata import ARCHITECTURE_KLEIN_9B, ARCHITECTURE_KLEIN_9B_FULL
from fizgig.utils.device import synchronize_device
from fizgig.utils.safetensors import (
    MemoryEfficientSafeOpen,
    TensorWeightAdapter,
    WeightTransformHooks,
    get_split_weight_filenames,
    load_split_weights,
)
from fizgig.modules.fp8 import apply_fp8_monkey_patch, load_safetensors_with_fp8_optimization

logger = logging.getLogger(__name__)


# region Model Info

@dataclass(frozen=True)
class KleinModelInfo:
    """Configuration for a Klein model variant."""
    guidance_distilled: bool
    architecture: str
    architecture_full: str
    defaults: dict
    fixed_params: set


# Klein 9B model variants (Distilled for fast sampling, Base for training)
KLEIN_MODEL_INFO = {
    "klein-9b": KleinModelInfo(
        guidance_distilled=True,
        architecture=ARCHITECTURE_KLEIN_9B,
        architecture_full=ARCHITECTURE_KLEIN_9B_FULL,
        defaults={"guidance": 1.0, "num_steps": 4},
        fixed_params={"guidance", "num_steps"},
    ),
    "klein-base-9b": KleinModelInfo(
        guidance_distilled=False,
        architecture=ARCHITECTURE_KLEIN_9B,
        architecture_full=ARCHITECTURE_KLEIN_9B_FULL,
        defaults={"guidance": 3.5, "num_steps": 20},
        fixed_params=set(),
    ),
}


def add_model_version_args(parser: argparse.ArgumentParser):
    """Add --model_version argument to parser."""
    choices = list(KLEIN_MODEL_INFO.keys())
    parser.add_argument("--model_version", type=str, default="klein-base-9b", choices=choices, help="Klein 9B model variant")


# endregion


# region Inference utilities

def generalized_time_snr_shift(t: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def get_schedule(num_steps: int, image_seq_len: int, flow_shift: Optional[float] = None) -> list[float]:
    """Compute timestep schedule for sampling."""
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if flow_shift is not None:
        timesteps = (timesteps * flow_shift) / (1 + (flow_shift - 1) * timesteps)
    else:
        timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps.tolist()


def get_simple_euler_schedule(num_steps: int, shift: float = 2.02) -> list[float]:
    """Replicate ComfyUI's Euler Simple schedule for Flux2/Klein.

    Matches ComfyUI's ModelSamplingFlux(shift) + simple_scheduler exactly:
    1. Pre-compute 10000 sigmas via flux_time_shift(shift, 1.0, t)
    2. Pick num_steps evenly-spaced points from the end
    3. Append 0.0

    This is resolution-independent (shift is fixed, not dynamic).
    """
    n_sigmas = 10000
    # Build the full sigma schedule (same as ModelSamplingFlux.set_parameters)
    t = torch.arange(1, n_sigmas + 1, dtype=torch.float64) / n_sigmas
    sigmas = torch.tensor([
        math.exp(shift) / (math.exp(shift) + (1.0 / ti - 1.0) ** 1.0)
        for ti in t.tolist()
    ])
    # simple_scheduler: pick evenly-spaced from the end
    ss = n_sigmas / num_steps
    schedule = [float(sigmas[-(1 + int(x * ss))]) for x in range(num_steps)]
    schedule.append(0.0)
    return schedule


def denoise(model, img, img_ids, txt, txt_ids, timesteps, guidance,
            img_cond_seq=None, img_cond_seq_ids=None):
    """Non-CFG denoising loop."""
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(tqdm(timesteps[:-1]), timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids
        if img_cond_seq is not None:
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

        with torch.no_grad(), torch.autocast(device_type=img.device.type, dtype=img.dtype):
            pred = model(x=img_input, x_ids=img_input_ids, timesteps=t_vec, ctx=txt, ctx_ids=txt_ids, guidance=guidance_vec)

        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]
        img = img + (t_prev - t_curr) * pred
    return img


def denoise_cfg(model, img, img_ids, txt, txt_ids, uncond_txt, uncond_txt_ids,
                timesteps, guidance, img_cond_seq=None, img_cond_seq_ids=None):
    """CFG denoising loop."""
    for t_curr, t_prev in zip(tqdm(timesteps[:-1]), timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        img_input = img
        img_input_ids = img_ids
        if img_cond_seq is not None:
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

        with torch.no_grad(), torch.autocast(device_type=img.device.type, dtype=img.dtype):
            pred_cond = model(x=img_input, x_ids=img_input_ids, timesteps=t_vec, ctx=txt, ctx_ids=txt_ids, guidance=None)
            pred_uncond = model(
                x=img_input, x_ids=img_input_ids, timesteps=t_vec, ctx=uncond_txt, ctx_ids=uncond_txt_ids, guidance=None
            )

        if img_cond_seq is not None:
            pred_cond = pred_cond[:, : img.shape[1]]
            pred_uncond = pred_uncond[:, : img.shape[1]]

        pred = pred_uncond + guidance * (pred_cond - pred_uncond)
        img = img + (t_prev - t_curr) * pred
    return img


# endregion


# region Model loading

def load_dit(
    device: Union[str, torch.device],
    model_version_info: KleinModelInfo,
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype] = None,
    fp8_scaled: bool = False,
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[list[float]] = None,
    disable_numpy_memmap: bool = False,
):
    """Load the Klein 9B DiT model.

    Args:
        device: Computation device (GPU).
        model_version_info: Klein model variant info.
        dit_path: Path to DiT safetensors file.
        attn_mode: Attention implementation (sdpa, flash, xformers, sage).
        split_attn: Whether to use split attention.
        loading_device: Device to load weights onto.
        dit_weight_dtype: Weight dtype (None for fp8_scaled).
        fp8_scaled: Enable FP8 scaled optimization.
        lora_weights_list: LoRA weights to merge during loading.
        lora_multipliers: LoRA multipliers.
        disable_numpy_memmap: Disable numpy memmap.

    Returns:
        KleinDiT model instance.
    """
    # Import here to avoid circular imports
    from fizgig.klein.model import KleinDiT, Klein9BParams, FP8_OPTIMIZATION_TARGET_KEYS, FP8_OPTIMIZATION_EXCLUDE_KEYS

    assert (not fp8_scaled and dit_weight_dtype is not None) or (fp8_scaled and dit_weight_dtype is None)

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    with init_empty_weights():
        params = Klein9BParams()
        model = KleinDiT(params, attn_mode, split_attn)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    logger.info(f"Loading DiT model from {dit_path}, device={loading_device}")
    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=dit_weight_dtype,
        target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
        disable_numpy_memmap=disable_numpy_memmap,
    )

    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)
        if loading_device.type != "cpu":
            logger.info(f"Moving weights to {loading_device}")
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    # Pre-quantized fp8 models (e.g. Comfy-Org's flux-2-klein-9b-fp8) store
    # raw fp8 weights alongside per-Linear dequantization scales
    # (X.weight_scale, X.input_scale). Keep fp8 residency on GPU (~9 GB vs
    # ~18 GB dequantized) and apply the same monkey-patch used by the
    # fp8_scaled path — the patched Linear forward does fp8 → bf16 dequant
    # per matmul using scale_weight, then the bf16 temporary is freed.
    # Memory budget: DiT stays at ~9 GB on GPU, 16 GB cards can run without
    # block swap.
    weight_scale_keys = [k for k in sd if k.endswith(".weight_scale")]
    if weight_scale_keys and not fp8_scaled:
        # Rename suffix to match apply_fp8_monkey_patch's convention
        # (.scale_weight is what it scans for and registers as a buffer).
        # Cast scale to bf16 to match the existing fp8_scaled path's dtype
        # convention — the patched forward multiplies fp8-cast-to-bf16 by
        # this, so keeping both as bf16 avoids fp32 upcasts inside F.linear.
        for ws_key in list(sd.keys()):
            if ws_key.endswith(".weight_scale"):
                new_key = ws_key[: -len(".weight_scale")] + ".scale_weight"
                sd[new_key] = sd.pop(ws_key).to(torch.bfloat16)
        # Input-scale keys aren't used by the non-scaled-mm forward path
        # (max_value=None in fp8_linear_forward_patch → input stays bf16).
        in_scale_keys = [k for k in sd if k.endswith(".input_scale")]
        for k in in_scale_keys:
            del sd[k]
        logger.info(
            f"Pre-quantized fp8 passthrough: keeping fp8 residency for "
            f"{len(weight_scale_keys)} Linears (~9 GB DiT vs ~18 GB bf16)."
        )
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)

    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded KleinDiT: {info}")
    return model


def load_vae(
    ckpt_path: str, dtype: torch.dtype, device: Union[str, torch.device], disable_mmap: bool = False
):
    """Load the AutoEncoder (VAE) for Klein."""
    from fizgig.klein.model import AutoEncoder, AutoEncoderParams

    logger.info("Building AutoEncoder")
    with init_empty_weights():
        ae = AutoEncoder(AutoEncoderParams()).to(dtype)

    logger.info(f"Loading VAE state dict from {ckpt_path}")
    sd = load_split_weights(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    info = ae.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded VAE: {info}")
    return ae


def load_text_encoder(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
):
    """Load the Qwen3-8B text encoder for Klein 9B."""
    from fizgig.klein.embedder import Qwen3Embedder, load_qwen3

    tokenizer, qwen3 = load_qwen3(
        ckpt_path, dtype, device, disable_mmap, state_dict,
        tokenizer_id="Qwen/Qwen3-8B",
    )
    return Qwen3Embedder(tokenizer, qwen3)


# endregion


# region LoRA + FP8 loading utilities

def detect_network_type(lora_sd: Dict[str, torch.Tensor]) -> str:
    """Detect network type from state dict keys."""
    for key in lora_sd:
        if "lora_down" in key:
            return "lora"
        if "hada_w1_a" in key:
            return "loha"
        if "lokr_w1" in key:
            return "lokr"
    return "lora"


def load_safetensors_with_lora_and_fp8(
    model_files: Union[str, List[str]],
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]],
    lora_multipliers: Optional[List[float]],
    fp8_optimization: bool,
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    target_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> Dict[str, torch.Tensor]:
    """Load model weights with optional LoRA merging and FP8 optimization."""
    if isinstance(model_files, str):
        model_files = [model_files]

    # Expand split weight files
    expanded = []
    for f in model_files:
        split = get_split_weight_filenames(f)
        expanded.extend(split if split else [f])
    model_files = expanded
    logger.info(f"Loading model files: {model_files}")

    # Build LoRA merge hook
    weight_hook = None
    if lora_weights_list and len(lora_weights_list) > 0:
        if lora_multipliers is None:
            lora_multipliers = [1.0] * len(lora_weights_list)
        while len(lora_multipliers) < len(lora_weights_list):
            lora_multipliers.append(1.0)
        lora_multipliers = lora_multipliers[:len(lora_weights_list)]

        list_of_lora_keys = [set(sd.keys()) for sd in lora_weights_list]
        lora_net_types = [detect_network_type(sd) for sd in lora_weights_list]

        logger.info(f"Merging LoRA weights. multipliers: {lora_multipliers}, types: {lora_net_types}")

        def weight_hook_func(model_key, model_weight: torch.Tensor, keep_on_calc_device=False):
            if not model_key.endswith(".weight"):
                return model_weight

            original_device = model_weight.device
            if original_device != calc_device:
                model_weight = model_weight.to(calc_device)

            for lora_keys, lora_sd, mult, net_type in zip(
                list_of_lora_keys, lora_weights_list, lora_multipliers, lora_net_types
            ):
                lora_name = "lora_unet_" + model_key.rsplit(".", 1)[0].replace(".", "_")

                if net_type != "lora":
                    continue  # Only standard LoRA supported in Fizgig for now

                down_key = lora_name + ".lora_down.weight"
                up_key = lora_name + ".lora_up.weight"
                alpha_key = lora_name + ".alpha"
                if down_key not in lora_keys or up_key not in lora_keys:
                    continue

                down_w = lora_sd[down_key].to(calc_device)
                up_w = lora_sd[up_key].to(calc_device)
                dim = down_w.size(0)
                alpha = lora_sd.get(alpha_key, dim)
                scale = alpha / dim

                original_dtype = model_weight.dtype
                if original_dtype.itemsize == 1:  # fp8
                    model_weight = model_weight.to(torch.float16)
                    down_w = down_w.to(torch.float16)
                    up_w = up_w.to(torch.float16)

                if len(model_weight.size()) == 2:
                    if len(up_w.size()) == 4:
                        up_w = up_w.squeeze(3).squeeze(2)
                        down_w = down_w.squeeze(3).squeeze(2)
                    model_weight = model_weight + mult * (up_w @ down_w) * scale
                elif down_w.size()[2:4] == (1, 1):
                    model_weight = (
                        model_weight
                        + mult * (up_w.squeeze(3).squeeze(2) @ down_w.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * scale
                    )
                else:
                    conved = torch.nn.functional.conv2d(down_w.permute(1, 0, 2, 3), up_w).permute(1, 0, 2, 3)
                    model_weight = model_weight + mult * conved * scale

                if original_dtype.itemsize == 1:
                    model_weight = model_weight.to(original_dtype)

                lora_keys.discard(down_key)
                lora_keys.discard(up_key)
                lora_keys.discard(alpha_key)

            if not keep_on_calc_device and original_device != calc_device:
                model_weight = model_weight.to(original_device)
            return model_weight

        weight_hook = weight_hook_func
    else:
        list_of_lora_keys = []

    # Load with or without FP8
    state_dict = _load_with_fp8_and_hook(
        model_files, fp8_optimization, calc_device, move_to_device,
        dit_weight_dtype, target_keys, exclude_keys,
        weight_hook=weight_hook,
        disable_numpy_memmap=disable_numpy_memmap,
        weight_transform_hooks=weight_transform_hooks,
    )

    for lora_keys in list_of_lora_keys:
        if lora_keys:
            logger.warning(f"Unused LoRA keys: {', '.join(lora_keys)}")

    return state_dict


def _load_with_fp8_and_hook(
    model_files, fp8_optimization, calc_device, move_to_device,
    dit_weight_dtype, target_keys, exclude_keys,
    weight_hook=None, disable_numpy_memmap=False, weight_transform_hooks=None,
) -> Dict[str, torch.Tensor]:
    """Internal: load safetensors with optional FP8 optimization and weight hooks."""
    if fp8_optimization:
        logger.info(f"Loading with FP8 optimization, hook={weight_hook is not None}")
        return load_safetensors_with_fp8_optimization(
            model_files, calc_device, target_keys, exclude_keys,
            move_to_device=move_to_device, weight_hook=weight_hook,
            disable_numpy_memmap=disable_numpy_memmap,
            weight_transform_hooks=weight_transform_hooks,
        )

    logger.info(f"Loading without FP8, dtype={dit_weight_dtype}, hook={weight_hook is not None}")
    state_dict = {}
    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
            f = TensorWeightAdapter(weight_transform_hooks, original_f) if weight_transform_hooks else original_f
            for key in tqdm(f.keys(), desc=f"Loading {os.path.basename(model_file)}", leave=False):
                if weight_hook is None and move_to_device:
                    value = f.get_tensor(key, device=calc_device, dtype=dit_weight_dtype)
                else:
                    value = f.get_tensor(key)
                    if weight_hook is not None:
                        value = weight_hook(key, value, keep_on_calc_device=move_to_device)
                    if move_to_device:
                        value = value.to(calc_device, dtype=dit_weight_dtype, non_blocking=True)
                    elif dit_weight_dtype is not None:
                        value = value.to(dit_weight_dtype)
                state_dict[key] = value

    if move_to_device:
        synchronize_device(calc_device)
    return state_dict

# endregion
