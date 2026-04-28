"""Standalone inference pipeline for Klein 9B.

Supports both Base (full-step CFG) and Distilled (4-step, no CFG) models.
Designed to be used independently of the training loop — by the GUI for
previews, by the profiler for diagnostics, and by future tools.

Usage:
    pipeline = KleinInferencePipeline()
    pipeline.load_models(dit_path, vae_path, text_encoder_path, device="cuda")
    image = pipeline.generate("a photo of a cat", width=1024, height=1024)
    pipeline.unload_models()
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from fizgig.klein.embedder import Qwen3Embedder, load_qwen3
from fizgig.klein.model import KleinDiT, Klein9BParams, AutoEncoder, AutoEncoderParams
from fizgig.klein.model_utils import (
    KLEIN_MODEL_INFO,
    KleinModelInfo,
    denoise,
    denoise_cfg,
    get_schedule,
    load_dit,
    load_vae,
)
from fizgig.klein.position import prc_img, prc_txt, scatter_ids, pack_control_latent
from fizgig.utils.device import clean_memory_on_device

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for a single generation.

    When values are None, defaults are derived from the loaded model:
      Distilled: 4 steps, guidance=1.0, no CFG
      Base: 50 steps, guidance=4.0, CFG=3.5
    """
    prompt: str = ""
    negative_prompt: str = " "
    width: int = 1024
    height: int = 1024
    num_steps: Optional[int] = None  # None = model default (4 Distilled, 50 Base)
    guidance_scale: Optional[float] = None  # None = model default (1.0 Distilled, 4.0 Base)
    cfg_scale: Optional[float] = None  # None = same as guidance_scale
    flow_shift: float = 3.0  # BFL official Klein shift
    seed: Optional[int] = None
    lora_path: Optional[str] = None
    lora_multiplier: float = 1.0


@dataclass
class AttentionMaps:
    """Container for captured attention maps from a generation run."""
    # Maps are stored per step, per block: maps[step_idx][block_name] = tensor
    # Each tensor is the text->patch attention submatrix
    maps: dict = field(default_factory=dict)
    enabled: bool = False
    steps_captured: int = 0


class KleinInferencePipeline:
    """Standalone inference pipeline for Klein 9B Base and Distilled.

    Manages model loading/unloading, text encoding, denoising, and VAE decoding.
    Supports LoRA injection and attention map capture.
    """

    def __init__(self):
        self.dit: Optional[KleinDiT] = None
        self.vae: Optional[AutoEncoder] = None
        self.text_encoder: Optional[Qwen3Embedder] = None
        self.model_info: Optional[KleinModelInfo] = None
        self.device: Optional[torch.device] = None
        self.dit_dtype: torch.dtype = torch.bfloat16

        # LoRA profiling state
        self._lora_network = None
        self._profiling_hooks = []
        self._profiling_buffer = {}

        # Extraction state
        self._extraction_accum = {}
        self._extraction_xx = {}
        self._extraction_sample_count = {}
        self._extraction_original_forwards = {}

        # Attention map capture state
        self._attn_hooks = []
        self._attn_maps = AttentionMaps()

    @property
    def is_loaded(self) -> bool:
        return self.dit is not None

    @property
    def is_distilled(self) -> bool:
        return self.model_info is not None and self.model_info.guidance_distilled

    # region Model Loading

    def load_models(
        self,
        dit_path: str,
        vae_path: str,
        text_encoder_path: str,
        model_version: str = "klein-base-9b",
        device: Union[str, torch.device] = "cuda",
        fp8_scaled: bool = False,
        fp8_text_encoder: bool = False,
        attn_mode: str = "torch",
        blocks_to_swap: int = 0,
    ):
        """Load all models for inference.

        Args:
            dit_path: Path to Klein 9B DiT checkpoint.
            vae_path: Path to VAE/AE checkpoint.
            text_encoder_path: Path to Qwen3-8B checkpoint.
            model_version: 'klein-9b' (Distilled) or 'klein-base-9b' (Base).
            device: Target device.
            fp8_scaled: Use FP8 scaled optimization for DiT.
            fp8_text_encoder: Use FP8 for text encoder.
            attn_mode: Attention implementation (torch, flash, xformers, sageattn).
            blocks_to_swap: Number of DiT blocks to swap to CPU (VRAM saving).
        """
        self.device = torch.device(device)
        self.model_info = KLEIN_MODEL_INFO[model_version]

        logger.info(f"Loading Klein 9B ({model_version}) for inference")

        # Load DiT
        dit_weight_dtype = None if fp8_scaled else torch.bfloat16
        self.dit = load_dit(
            device=self.device,
            model_version_info=self.model_info,
            dit_path=dit_path,
            attn_mode=attn_mode,
            split_attn=False,
            loading_device=self.device,
            dit_weight_dtype=dit_weight_dtype,
            fp8_scaled=fp8_scaled,
        )

        # Enable block swap if requested
        if blocks_to_swap > 0:
            self.dit.enable_block_swap(blocks_to_swap, device=self.device, supports_backward=False)
            self.dit.prepare_block_swap_before_forward()

        self.dit.eval()
        self.dit_dtype = torch.bfloat16

        # Load VAE
        self.vae = load_vae(vae_path, dtype=torch.bfloat16, device="cpu")
        self.vae.eval()

        # Load text encoder
        te_dtype = torch.float8_e4m3fn if fp8_text_encoder else torch.bfloat16
        tokenizer, qwen3 = load_qwen3(
            text_encoder_path, dtype=te_dtype, device=self.device,
            disable_mmap=True, tokenizer_id="Qwen/Qwen3-8B",
        )
        self.text_encoder = Qwen3Embedder(tokenizer, qwen3)

        logger.info("All models loaded for inference")

    def load_lora(self, lora_path: str, multiplier: float = 1.0):
        """Load and apply a LoRA to the current DiT model.

        Args:
            lora_path: Path to LoRA safetensors file.
            multiplier: LoRA weight multiplier.
        """
        if self.dit is None:
            raise RuntimeError("DiT not loaded. Call load_models() first.")

        from fizgig.networks.lora_klein import create_arch_network_from_weights
        from fizgig.networks.lora import ensure_kohya_lora_state_dict
        from safetensors.torch import load_file

        logger.info(f"Loading LoRA from {lora_path} (multiplier={multiplier})")
        weights_sd = load_file(lora_path)
        weights_sd = ensure_kohya_lora_state_dict(weights_sd)  # auto-convert PEFT → kohya
        network = create_arch_network_from_weights(
            multiplier=multiplier,
            weights_sd=weights_sd,
            unet=self.dit,
            for_inference=True,
        )
        network.merge_to(self.dit, weights_sd, self.dit_dtype, self.device)
        logger.info(f"LoRA applied: {len(weights_sd)} keys merged")

    def load_lora_for_profiling(self, lora_path: str, multiplier: float = 1.0):
        """Load a LoRA without merging — keeps it toggleable for differential profiling.

        Unlike load_lora() which permanently merges weights, this keeps the
        LoRA network separate so it can be enabled/disabled at runtime via
        enable_lora() / disable_lora().
        """
        if self.dit is None:
            raise RuntimeError("DiT not loaded. Call load_models() first.")

        from fizgig.networks.lora_klein import create_arch_network_from_weights
        from safetensors.torch import load_file

        logger.info(f"Loading LoRA for profiling from {lora_path} (multiplier={multiplier})")
        weights_sd = load_file(lora_path)

        # Auto-convert PEFT/diffusers-format LoRAs (diffusion_model.*.lora_A/B.weight)
        # to Fizgig/kohya format in-memory. Kohya-format files pass through unchanged.
        from fizgig.networks.lora import ensure_kohya_lora_state_dict
        weights_sd = ensure_kohya_lora_state_dict(weights_sd)

        network = create_arch_network_from_weights(
            multiplier=multiplier,
            weights_sd=weights_sd,
            unet=self.dit,
            for_inference=True,
        )

        # Apply hooks (replaces forward methods) but don't merge weights
        network.apply_to(text_encoders=None, unet=self.dit, apply_text_encoder=False, apply_unet=True)

        # Load the actual LoRA weights into the network modules
        info = network.load_state_dict(weights_sd, strict=False)
        logger.info(f"LoRA loaded for profiling: {len(network.unet_loras)} modules, enabled=True")

        # Move LoRA weights to device
        network.to(self.device, dtype=self.dit_dtype)

        self._lora_network = network

    def enable_lora(self):
        """Enable the profiling LoRA (forward passes will include LoRA contribution)."""
        if self._lora_network is None:
            raise RuntimeError("No profiling LoRA loaded. Call load_lora_for_profiling() first.")
        self._lora_network.set_enabled(True)

    def disable_lora(self):
        """Disable the profiling LoRA (forward passes will be base model only)."""
        if self._lora_network is None:
            raise RuntimeError("No profiling LoRA loaded.")
        self._lora_network.set_enabled(False)

    # Klein 9B empirical block patterns (refined from user's ComfyUI testing)
    # Each category is a list of (regex pattern, multiplier) tuples.
    # The multiplier is applied during extraction/use. Single 2 appears in
    # style_composition at 0.5 strength as a bridge block.
    BLOCK_PATTERNS = {
        "style_composition": [
            (r"double_blocks_\d_", 1.0),         # double 0-7, full
            (r"single_blocks_[01]_", 1.0),       # single 0-1, full
            (r"single_blocks_2_", 0.5),          # single 2, half strength (bridge)
        ],
        "identity": [
            (r"single_blocks_(1[0-6]|[1-9])_", 1.0),  # single 1-16, full (overlaps style_composition at 1 and details at 12-16)
        ],
        "details": [
            (r"single_blocks_(1[2-9]|2[0-3])_", 1.0),  # single 12-23, full
        ],
    }

    def enable_blocks(self, categories: list):
        """Enable LoRA modules in the given block categories with their preset multipliers.

        Args:
            categories: list of 'style_composition', 'identity', 'details', or 'all'.
        """
        if self._lora_network is None:
            raise RuntimeError("No LoRA loaded.")

        if "all" in categories or not categories:
            self._lora_network.set_enabled(True)
            # Reset multipliers to 1.0 (in case they were set by a previous call)
            for lora in self._lora_network.unet_loras:
                lora.multiplier = 1.0
            return

        # First disable all and reset multipliers
        self._lora_network.set_enabled(False)
        for lora in self._lora_network.unet_loras:
            lora.multiplier = 1.0

        # Apply each category's patterns with their multipliers
        for cat in categories:
            if cat in self.BLOCK_PATTERNS:
                for pattern, mult in self.BLOCK_PATTERNS[cat]:
                    self._lora_network.set_module_enabled_by_pattern(pattern, True)
                    self._lora_network.set_module_multiplier_by_pattern(pattern, mult)

    @property
    def has_profiling_lora(self) -> bool:
        return self._lora_network is not None

    # endregion

    # region Profiling Hooks

    def setup_profiling_hooks(self):
        """Register hooks that capture per-block output norms into a buffer.

        Call clear_profiling_buffer() before each forward pass, then read
        get_profiling_buffer() after.
        """
        self._remove_profiling_hooks()
        self._profiling_hooks = []
        self._profiling_buffer = {}

        for i, block in enumerate(self.dit.double_blocks):
            name = f"double_blocks.{i}"
            hook = self._make_profiling_hook(name, is_double=True)
            handle = block.register_forward_hook(hook)
            self._profiling_hooks.append(handle)

        for i, block in enumerate(self.dit.single_blocks):
            name = f"single_blocks.{i}"
            hook = self._make_profiling_hook(name, is_double=False)
            handle = block.register_forward_hook(hook)
            self._profiling_hooks.append(handle)

        logger.info(f"Profiling hooks registered: {len(self._profiling_hooks)} blocks")

    def _make_profiling_hook(self, block_name: str, is_double: bool):
        """Create a forward hook that stores output norm for this block."""
        buf = self._profiling_buffer

        def hook(module, input, output):
            if is_double:
                # DoubleStreamBlock returns (img, txt)
                img_out, txt_out = output
                buf[block_name] = img_out.detach().float().norm().item()
            else:
                # SingleStreamBlock returns a tensor
                buf[block_name] = output.detach().float().norm().item()
        return hook

    def clear_profiling_buffer(self):
        """Clear the profiling buffer before a forward pass."""
        if hasattr(self, '_profiling_buffer'):
            self._profiling_buffer.clear()

    def get_profiling_buffer(self) -> dict:
        """Get the profiling buffer after a forward pass. Keys are block names, values are output norms."""
        return dict(self._profiling_buffer) if hasattr(self, '_profiling_buffer') else {}

    def _remove_profiling_hooks(self):
        """Remove all profiling hooks."""
        if hasattr(self, '_profiling_hooks'):
            for handle in self._profiling_hooks:
                handle.remove()
            self._profiling_hooks.clear()

    # endregion

    # region Extraction Capture (per-layer input x and delta d)

    def setup_extraction_capture(self):
        """Patch LoRAInfModule.forward to accumulate D @ X^T per module.

        For each enabled LoRA module, we compute:
            x = input to the Linear layer
            d = lora_up(lora_down(x)) * scale * multiplier (the LoRA's contribution)
            C += d.T @ x  (accumulator, shape: [in_dim, out_dim] stored transposed)

        Actually we accumulate D^T @ X where D is (batch*seq, out_dim), X is (batch*seq, in_dim),
        so D^T @ X has shape (out_dim, in_dim) — the cross-correlation matrix.
        """
        if self._lora_network is None:
            raise RuntimeError("No LoRA loaded. Call load_lora_for_profiling() first.")

        self._extraction_accum = {}  # lora_name -> C_dx = D^T @ X (out_dim, in_dim)
        self._extraction_xx = {}  # lora_name -> C_xx = X^T @ X (in_dim, in_dim) input covariance
        self._extraction_sample_count = {}  # lora_name -> number of tokens accumulated
        self._extraction_original_forwards = {}  # lora_name -> original forward method

        for lora in self._lora_network.unet_loras:
            if not lora.enabled:
                continue
            # Patch org_module.forward to capture x and delta
            self._patch_lora_forward(lora)

    def _patch_lora_forward(self, lora):
        """Replace the org_module's forward with one that captures activations.

        NOTE: LoRA's apply_to() replaced org_module.forward with lora.forward (bound method).
        So to intercept actual forward calls during model inference, we must patch
        org_module.forward directly, not lora.forward.
        """
        installed_forward = lora.org_module_ref[0].forward
        lora_name = lora.lora_name
        accum = self._extraction_accum
        accum_xx = self._extraction_xx
        counts = self._extraction_sample_count

        def capturing_forward(x):
            y = installed_forward(x)

            if lora.enabled:
                # compute_delta is defined uniformly on every LoRA variant
                # (standard LoRA, LoKR, LoHa) so the capture hook is
                # variant-agnostic.
                delta = lora.compute_delta(x)

                # Flatten to (N, in_dim) and (N, out_dim) in float32
                x_flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
                d_flat = delta.reshape(-1, delta.shape[-1]).to(torch.float32)

                # Accumulate C_dx = D^T @ X (shape: out_dim, in_dim)
                c_dx = d_flat.t() @ x_flat
                # Accumulate DIAGONAL of C_xx = sum(X^2) per dim (shape: in_dim,)
                # This is a per-dimension input variance proxy, memory-efficient
                # (full C_xx would be in_dim × in_dim which is prohibitive for linear2)
                c_xx_diag = (x_flat * x_flat).sum(dim=0)

                if lora_name not in accum:
                    accum[lora_name] = c_dx.cpu()
                    accum_xx[lora_name] = c_xx_diag.cpu()
                    counts[lora_name] = x_flat.shape[0]
                else:
                    accum[lora_name] = accum[lora_name] + c_dx.cpu()
                    accum_xx[lora_name] = accum_xx[lora_name] + c_xx_diag.cpu()
                    counts[lora_name] += x_flat.shape[0]

            return y

        # Patch the org_module's forward (the one actually called during inference)
        lora.org_module_ref[0].forward = capturing_forward
        # Track for restoration
        self._extraction_original_forwards[lora.lora_name] = installed_forward

    def clear_extraction_data(self):
        """Clear accumulated extraction data (between independent runs)."""
        if hasattr(self, '_extraction_accum'):
            self._extraction_accum.clear()
            self._extraction_xx.clear()
            self._extraction_sample_count.clear()

    def get_extraction_data(self) -> dict:
        """Return accumulated C_dx and C_xx per lora_name.

        Returns:
            {lora_name: {
                "c_dx": tensor(out_dim, in_dim),  # D^T @ X cross-covariance
                "c_xx": tensor(in_dim, in_dim),   # X^T @ X input covariance
                "samples": int,
            }}
        """
        result = {}
        for name, accum in self._extraction_accum.items():
            result[name] = {
                "c_dx": accum,
                "c_xx": self._extraction_xx.get(name),
                "samples": self._extraction_sample_count.get(name, 0),
            }
        return result

    def remove_extraction_capture(self):
        """Restore original forward methods on LoRA modules."""
        if not hasattr(self, '_extraction_original_forwards'):
            return
        for lora in self._lora_network.unet_loras:
            if lora.lora_name in self._extraction_original_forwards:
                lora.org_module_ref[0].forward = self._extraction_original_forwards[lora.lora_name]
        self._extraction_original_forwards.clear()

    # endregion

    def unload_models(self):
        """Unload all models and free VRAM."""
        self._remove_attn_hooks()
        self._remove_profiling_hooks()
        if hasattr(self, '_lora_network') and self._lora_network is not None:
            del self._lora_network
            self._lora_network = None

        if self.dit is not None:
            del self.dit
            self.dit = None
        if self.vae is not None:
            del self.vae
            self.vae = None
        if self.text_encoder is not None:
            del self.text_encoder
            self.text_encoder = None

        clean_memory_on_device(self.device)
        logger.info("Models unloaded")

    # endregion

    # region Text Encoding

    def encode_prompt(
        self, prompt: str, negative_prompt: str = " "
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a prompt and negative prompt with Qwen3.

        Returns:
            Tuple of (ctx_vec, negative_ctx_vec), each (1, 512, 12288).
        """
        if self.text_encoder is None:
            raise RuntimeError("Text encoder not loaded.")

        te_dtype = self.text_encoder.dtype
        autocast_dtype = torch.bfloat16 if te_dtype.itemsize == 1 else te_dtype

        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype), torch.no_grad():
            ctx_vec = self.text_encoder([prompt])
            neg_ctx_vec = self.text_encoder([negative_prompt])

        return ctx_vec, neg_ctx_vec

    def unload_text_encoder(self):
        """Move text encoder to CPU to free VRAM for generation."""
        if self.text_encoder is not None:
            self.text_encoder.to("cpu")
            clean_memory_on_device(self.device)

    def reload_text_encoder(self):
        """Move text encoder back to GPU."""
        if self.text_encoder is not None:
            self.text_encoder.to(self.device)

    # endregion

    # region Generation

    def generate(self, config: GenerationConfig) -> Image.Image:
        """Generate a single image from a GenerationConfig.

        This is the high-level API. For more control, use encode_prompt()
        and generate_from_embeddings() directly.

        Returns:
            PIL Image.
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Encode text
        ctx_vec, neg_ctx_vec = self.encode_prompt(config.prompt, config.negative_prompt)

        # Free text encoder VRAM for generation
        self.unload_text_encoder()

        # Generate
        image = self.generate_from_embeddings(
            ctx_vec=ctx_vec,
            negative_ctx_vec=neg_ctx_vec,
            width=config.width,
            height=config.height,
            num_steps=config.num_steps,
            guidance_scale=config.guidance_scale,
            cfg_scale=config.cfg_scale,
            flow_shift=config.flow_shift,
            seed=config.seed,
        )

        return image

    def generate_from_embeddings(
        self,
        ctx_vec: torch.Tensor,
        negative_ctx_vec: Optional[torch.Tensor] = None,
        width: int = 1024,
        height: int = 1024,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        cfg_scale: Optional[float] = None,
        flow_shift: Optional[float] = None,
        seed: Optional[int] = None,
        capture_attention: bool = False,
    ) -> Image.Image:
        """Generate an image from pre-encoded text embeddings.

        Args:
            ctx_vec: Text embeddings (1, 512, 12288).
            negative_ctx_vec: Negative text embeddings (for Base CFG).
            width: Output width (will be rounded to 16).
            height: Output height (will be rounded to 16).
            num_steps: Denoising steps (default: 4 for Distilled, 28 for Base).
            guidance_scale: Guidance scale (default: 1.0 for Distilled, 3.5 for Base).
            cfg_scale: CFG scale for Base (default: same as guidance_scale).
            flow_shift: Flow shift (None = automatic empirical mu).
            seed: Random seed.
            capture_attention: Enable attention map capture for this generation.

        Returns:
            PIL Image.
        """
        if self.dit is None or self.vae is None:
            raise RuntimeError("DiT and VAE must be loaded.")

        # Apply model-variant defaults for any None values
        defaults = self.model_info.defaults
        if num_steps is None:
            num_steps = int(defaults.get("num_steps", 20))
        if guidance_scale is None:
            guidance_scale = float(defaults.get("guidance", 1.0))
        if cfg_scale is None:
            cfg_scale = guidance_scale

        # Round dimensions to 16
        width = (width // 16) * 16
        height = (height // 16) * 16

        device = self.device

        # Prepare text embeddings
        ctx = ctx_vec.to(device=device, dtype=torch.bfloat16)
        ctx, ctx_ids = prc_txt(ctx)

        if negative_ctx_vec is not None:
            neg_ctx = negative_ctx_vec.to(device=device, dtype=torch.bfloat16)
            neg_ctx, neg_ctx_ids = prc_txt(neg_ctx)
        else:
            neg_ctx, neg_ctx_ids = None, None

        # Generate initial latent noise
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        packed_h, packed_w = height // 16, width // 16
        latents = randn_tensor(
            (1, 128, packed_h, packed_w),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        x, x_ids = prc_img(latents)

        # Setup attention capture if requested
        if capture_attention:
            self._setup_attn_hooks()

        # Prepare block swap for forward-only inference
        if hasattr(self.dit, '_offloader') and self.dit._offloader is not None:
            self.dit._offloader.set_forward_only(True)
            self.dit.prepare_block_swap_before_forward()

        # Compute timestep schedule
        timesteps = get_schedule(num_steps, x.shape[1], flow_shift)

        # Denoise — Distilled uses denoise() (no CFG), Base uses denoise_cfg()
        if self.is_distilled:
            x = denoise(
                self.dit, x, x_ids, ctx, ctx_ids,
                timesteps=timesteps, guidance=guidance_scale,
            )
        else:
            if neg_ctx is None:
                raise ValueError("Base model requires negative_ctx_vec for CFG")
            x = denoise_cfg(
                self.dit, x, x_ids, ctx, ctx_ids, neg_ctx, neg_ctx_ids,
                timesteps=timesteps, guidance=cfg_scale,
            )

        # Unpack to spatial
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)  # B, C, H, W
        latent = x.to(self.vae.dtype)
        del x

        # Decode with VAE
        self.vae.to(device)
        self.vae.eval()
        with torch.no_grad():
            pixels = self.vae.decode(latent)
        del latent

        pixels = pixels.to(torch.float32).cpu()
        pixels = (pixels / 2 + 0.5).clamp(0, 1)

        self.vae.to("cpu")
        clean_memory_on_device(device)

        # Convert to PIL
        img_np = (pixels[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np)

    # endregion

    # region Attention Map Capture

    def enable_attention_capture(self):
        """Enable attention map capture for subsequent generations."""
        self._attn_maps = AttentionMaps(enabled=True)
        self._setup_attn_hooks()

    def disable_attention_capture(self):
        """Disable attention map capture and remove hooks."""
        self._attn_maps.enabled = False
        self._remove_attn_hooks()

    def get_attention_maps(self) -> AttentionMaps:
        """Retrieve captured attention maps from the last generation."""
        return self._attn_maps

    def _setup_attn_hooks(self):
        """Register forward hooks on attention layers to capture attention weights."""
        self._remove_attn_hooks()  # Clear any existing hooks

        if self.dit is None:
            return

        self._attn_maps = AttentionMaps(enabled=True)

        # Hook into DoubleStreamBlock attention (text <-> image cross-attention)
        for i, block in enumerate(self.dit.double_blocks):
            hook = self._make_double_block_hook(f"double_blocks.{i}")
            handle = block.register_forward_hook(hook)
            self._attn_hooks.append(handle)

        # Hook into SingleStreamBlock attention (joint self-attention)
        for i, block in enumerate(self.dit.single_blocks):
            hook = self._make_single_block_hook(f"single_blocks.{i}")
            handle = block.register_forward_hook(hook)
            self._attn_hooks.append(handle)

        logger.info(f"Attention capture hooks registered: {len(self._attn_hooks)} blocks")

    def _remove_attn_hooks(self):
        """Remove all registered attention hooks."""
        for handle in self._attn_hooks:
            handle.remove()
        self._attn_hooks.clear()

    def _make_double_block_hook(self, block_name: str):
        """Create a forward hook for DoubleStreamBlock that captures attention patterns."""
        attn_maps = self._attn_maps

        def hook(module, input, output):
            # DoubleStreamBlock forward returns (img, txt) tuple
            # The attention weights are computed inside the block but not returned.
            # We record the output activation magnitudes per block as a proxy
            # for attention influence. Full attention weight capture requires
            # modifying the attention function (Phase 2 profiler work).
            if not attn_maps.enabled:
                return

            step_key = f"step_{attn_maps.steps_captured}"
            if step_key not in attn_maps.maps:
                attn_maps.maps[step_key] = {}

            img_out, txt_out = output
            attn_maps.maps[step_key][block_name] = {
                "img_norm": img_out.detach().float().norm(dim=-1).mean().item(),
                "txt_norm": txt_out.detach().float().norm(dim=-1).mean().item(),
            }

        return hook

    def _make_single_block_hook(self, block_name: str):
        """Create a forward hook for SingleStreamBlock that captures activation magnitudes."""
        attn_maps = self._attn_maps

        def hook(module, input, output):
            if not attn_maps.enabled:
                return

            step_key = f"step_{attn_maps.steps_captured}"
            if step_key not in attn_maps.maps:
                attn_maps.maps[step_key] = {}

            attn_maps.maps[step_key][block_name] = {
                "out_norm": output.detach().float().norm(dim=-1).mean().item(),
            }

        return hook

    def _increment_step_counter(self):
        """Call after each denoising step to advance the attention map step counter."""
        if self._attn_maps.enabled:
            self._attn_maps.steps_captured += 1

    # endregion
