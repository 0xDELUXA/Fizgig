"""RepairEngine — owns the Klein pipeline and primary/donor LoRA networks.

Single preview entry point: generate_preview(state) -> PIL.Image. v2 will
replace the body to consult an activation cache; the UI never reaches around
this method.
"""

import json
import logging
import os
import re
from typing import List, Optional, Set

from PIL import Image

from fizgig.repair_studio.state import SliderState, block_regex

logger = logging.getLogger(__name__)


def find_profile_for_hash(profiles_dir: str, target_hash: str) -> Optional[dict]:
    """Scan `profiles_dir` for *.json sidecars written by the Profiler and
    return the first payload whose `hash` field matches `target_hash`.

    Returns None if no directory, no match, or any parse error. Designed to
    be cheap enough to call from a GUI event handler — the directory is
    small (one JSON per profiled LoRA) and each JSON is tiny.
    """
    if not target_hash or not profiles_dir or not os.path.isdir(profiles_dir):
        return None
    try:
        candidates = sorted(fn for fn in os.listdir(profiles_dir) if fn.endswith(".json"))
    except Exception:
        return None
    for fn in candidates:
        path = os.path.join(profiles_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("hash") == target_hash:
            payload["_sidecar_path"] = path
            return payload
    return None


_LORA_BLOCK_RX = re.compile(r"(double|single)_blocks_(\d+)_")


def _extract_block_ids_from_network(network) -> Set[str]:
    """Return the set of block_ids (e.g. {'single_5', 'double_3'}) that have
    LoRA modules in this network. An Identity-only LoRA typically returns a
    subset like {'single_1'..'single_16'}; a full-model LoRA returns all 32.
    """
    ids: Set[str] = set()
    if network is None:
        return ids
    for mod in network.unet_loras:
        m = _LORA_BLOCK_RX.search(mod.lora_name)
        if m:
            ids.add(f"{m.group(1)}_{int(m.group(2))}")
    return ids


class RepairEngine:
    def __init__(self):
        self.pipeline = None
        self.primary_network = None
        self.donor_network = None
        self.primary_path: Optional[str] = None
        self.donor_path: Optional[str] = None

        # Set of block_ids present in each network. Lets the UI grey out
        # sliders for blocks the LoRA doesn't actually touch.
        self.primary_block_ids: Set[str] = set()
        self.donor_block_ids: Set[str] = set()

        # SHA-256 of the primary LoRA's bytes — used by the GUI to cross-link
        # to any matching Profiler report sidecar in profiles_dir.
        self.primary_hash: Optional[str] = None

        # v2 hook — set by UI before generate_preview.
        self._changed_blocks: set = set()

        # v2 Turbo Preview — activation cache
        self._turbo_enabled: bool = False
        self._act_cache: Optional[dict] = None       # timestep_idx → ActivationCacheEntry
        self._act_cache_key: Optional[tuple] = None   # (primary_path, donor_path, seed, prompt, w, h)

        # Cache the last-generated baseline keyed on (primary_path, seed,
        # prompt, w, h) — only regenerate baseline when these change.
        self._baseline_cache_key = None
        self._baseline_cache_image: Optional[Image.Image] = None

    # ------------------------------------------------------------------
    # Pipeline + LoRA loading
    # ------------------------------------------------------------------

    def ensure_pipeline(
        self,
        dit_path: str,
        vae_path: str,
        text_encoder_path: str,
        model_version: str = "klein-9b",
        device: str = "cuda",
        fp8_scaled: bool = False,
        fp8_text_encoder: bool = True,
        blocks_to_swap: int = 0,
    ) -> None:
        """Lazy-load the Klein pipeline. No-op if already loaded.

        fp8_text_encoder defaults to True — Qwen3-8B in bf16 is ~16GB; fp8 is ~8GB,
        keeping room for the bf16 DiT (~18GB) on a 32GB card.
        """
        if self.pipeline is not None and self.pipeline.is_loaded:
            return

        from fizgig.klein.inference import KleinInferencePipeline

        self.pipeline = KleinInferencePipeline()
        self.pipeline.load_models(
            dit_path=dit_path,
            vae_path=vae_path,
            text_encoder_path=text_encoder_path,
            model_version=model_version,
            device=device,
            fp8_scaled=fp8_scaled,
            fp8_text_encoder=fp8_text_encoder,
            blocks_to_swap=blocks_to_swap,
        )
        # Offload TE to CPU between calls (saves ~8GB). generate_preview
        # reloads it to GPU before each encode.
        try:
            self.pipeline.unload_text_encoder()
        except Exception:
            logger.exception("Failed to offload text encoder (non-fatal)")
        logger.info("Repair Studio pipeline ready (model_version=%s, fp8_te=%s)",
                    model_version, fp8_text_encoder)

    def load_primary(self, path: str) -> None:
        """Load primary LoRA without merging — keeps multipliers live.

        v1 limitation: changing primary requires reset() first. The DiT's
        forward methods are patched on apply_to(); we don't unwind those.
        """
        if self.pipeline is None or not self.pipeline.is_loaded:
            raise RuntimeError("Pipeline not loaded; call ensure_pipeline() first.")
        if self.primary_network is not None:
            raise RuntimeError("Primary already loaded — call reset() to swap.")

        self.pipeline.load_lora_for_profiling(path, multiplier=1.0)
        self.primary_network = self.pipeline._lora_network
        self.primary_path = path
        self.primary_block_ids = _extract_block_ids_from_network(self.primary_network)
        # Content hash for Profiler cross-link (GUI looks up the sidecar).
        from fizgig.profiler.visualize import compute_lora_hash
        self.primary_hash = compute_lora_hash(path)
        self._invalidate_baseline_cache()
        self._invalidate_activation_cache()
        logger.info("Repair Studio primary loaded: %s (blocks: %d, hash: %s)",
                    path, len(self.primary_block_ids),
                    self.primary_hash[:12] + "…" if self.primary_hash else "?")

    def load_donor(self, path: str) -> None:
        """Load a second LoRA on top (chained, additive). All blocks start disabled."""
        if self.pipeline is None or not self.pipeline.is_loaded:
            raise RuntimeError("Pipeline not loaded.")
        if self.primary_network is None:
            raise RuntimeError("Load primary LoRA before donor.")
        if self.donor_network is not None:
            raise RuntimeError("Donor already loaded — call reset() or unload_donor() first.")

        from fizgig.networks.lora_klein import create_arch_network_from_weights
        from fizgig.networks.lora import ensure_kohya_lora_state_dict
        from safetensors.torch import load_file

        weights_sd = load_file(path)
        weights_sd = ensure_kohya_lora_state_dict(weights_sd)

        network = create_arch_network_from_weights(
            multiplier=1.0,
            weights_sd=weights_sd,
            unet=self.pipeline.dit,
            for_inference=True,
        )
        # apply_to chains forwards on top of the primary patches.
        network.apply_to(
            text_encoders=None,
            unet=self.pipeline.dit,
            apply_text_encoder=False,
            apply_unet=True,
        )
        network.load_state_dict(weights_sd, strict=False)
        network.to(self.pipeline.device, dtype=self.pipeline.dit_dtype)

        # Disable all donor modules — they only contribute when the user
        # explicitly enables them per-block via the slider state.
        network.set_enabled(False)

        self.donor_network = network
        self.donor_path = path
        self.donor_block_ids = _extract_block_ids_from_network(network)
        self._invalidate_activation_cache()
        logger.info("Repair Studio donor loaded: %s (%d modules, %d blocks)",
                    path, len(network.unet_loras), len(self.donor_block_ids))

    def unload_donor(self) -> None:
        """Disable donor (modules stay patched but transparent). Cheap; no DiT reload."""
        if self.donor_network is not None:
            self.donor_network.set_enabled(False)
            self.donor_network = None
            self.donor_path = None
            self.donor_block_ids = set()
            self._invalidate_activation_cache()
            logger.info("Donor unloaded (chain remains; reset() for full cleanup)")

    def reset(self) -> None:
        """Full unload — drops pipeline, primary, donor. Next ensure_pipeline reloads from disk."""
        import gc
        import torch
        from fizgig.utils.device import clean_memory_on_device

        import gc
        import torch

        # Invalidate caches first (drops large GPU tensors from turbo cache).
        self._invalidate_baseline_cache()
        self._invalidate_activation_cache()

        # Drop LoRA networks BEFORE the pipeline — their forward-hook closures
        # capture `org_forward` (a bound method on the DiT's Linear layers),
        # creating circular references that prevent GC from freeing the DiT.
        # Explicitly clear the module lists to break the cycles.
        for net in (self.primary_network, self.donor_network):
            if net is not None:
                try:
                    for lora in net.unet_loras:
                        lora.org_forward = None
                    net.unet_loras.clear()
                    for lora in net.text_encoder_loras:
                        lora.org_forward = None
                    net.text_encoder_loras.clear()
                except Exception:
                    pass
        self.primary_network = None
        self.donor_network = None

        # Force GC to break any remaining ref cycles before pipeline unload.
        gc.collect()

        if self.pipeline is not None:
            try:
                self.pipeline.unload_models()
            except Exception:
                logger.exception("Pipeline unload raised; continuing")
        self.pipeline = None

        self.primary_path = None
        self.donor_path = None
        self.primary_block_ids = set()
        self.donor_block_ids = set()
        self.primary_hash = None
        self._changed_blocks.clear()

        # Final GC + CUDA cache flush.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Activation cache (v2 Turbo Preview)
    # ------------------------------------------------------------------

    def _invalidate_activation_cache(self):
        """Release all cached activation tensors."""
        self._act_cache = None
        self._act_cache_key = None

    def _compute_resume_point(self) -> Optional[tuple]:
        """From _changed_blocks, find the earliest block to resume from."""
        if not self._changed_blocks:
            return None
        doubles = sorted(int(b.split("_")[1]) for b in self._changed_blocks if b.startswith("double_"))
        singles = sorted(int(b.split("_")[1]) for b in self._changed_blocks if b.startswith("single_"))
        if doubles:
            return ("double", doubles[0])
        if singles:
            return ("single", singles[0])
        return None

    # ------------------------------------------------------------------
    # State application
    # ------------------------------------------------------------------

    def apply_state(self, state: SliderState) -> None:
        """Push the slider state into the live LoRA networks."""
        if self.primary_network is None:
            raise RuntimeError("No primary LoRA loaded.")

        for block_id, bs in state.blocks.items():
            pat = block_regex(block_id)
            self.primary_network.set_module_enabled_by_pattern(pat, bs.primary_enabled, target="unet")
            self.primary_network.set_module_multiplier_by_pattern(pat, bs.primary_strength, target="unet")
            if self.donor_network is not None:
                self.donor_network.set_module_enabled_by_pattern(pat, bs.donor_enabled, target="unet")
                self.donor_network.set_module_multiplier_by_pattern(pat, bs.donor_strength, target="unet")

    def mark_blocks_changed(self, block_ids: List[str]) -> None:
        """v2 hook — UI calls this BEFORE generate_preview with diffed block ids.
        v1: noop (kept on the engine, not on SliderState, so presets stay clean)."""
        self._changed_blocks.update(block_ids)

    # ------------------------------------------------------------------
    # Generation — the single preview entry point (v2 replaces body only)
    # ------------------------------------------------------------------

    def generate_preview(self, state: SliderState) -> Image.Image:
        """Apply state, run a 4-step Distilled generation, return PIL image.

        Inlines the proven training-loop sample path (trainer.do_inference)
        instead of pipeline.generate() — the latter is a dormant untested
        method that produces blocky noise in this call site. See trainer.py
        sample_image_inference / do_inference for the reference path.

        v1: full forward every call. v2 will diff against last state, reuse
        cached activations from before the earliest changed block, and clear
        self._changed_blocks at the end.
        """
        if self.pipeline is None or not self.pipeline.is_loaded:
            raise RuntimeError("Pipeline not loaded.")

        import os
        import numpy as np
        import torch
        from diffusers.utils.torch_utils import randn_tensor
        from fizgig.klein.model_utils import denoise, denoise_cfg, get_schedule
        from fizgig.klein.position import prc_img, prc_txt, scatter_ids
        from fizgig.utils.device import clean_memory_on_device

        # --- Diagnostic log for Distilled debugging ---
        log_lines = []
        def dlog(msg):
            logger.info("[REPAIR] %s", msg)
            log_lines.append(msg)

        # Phase boundary 1: release any leftover allocator state from the
        # previous preview before we spike for the TE encode. Critical on
        # 16 GB cards where fragmented buffers can push the encode peak
        # over the budget.
        clean_memory_on_device(self.pipeline.device)

        self.apply_state(state)

        pipeline = self.pipeline
        device = pipeline.device
        defaults = pipeline.model_info.defaults
        num_steps = int(defaults.get("num_steps", 4))
        guidance_scale = float(defaults.get("guidance", 1.0))
        cfg_scale = float(defaults.get("cfg_scale", guidance_scale))

        dlog(f"=== generate_preview ===")
        dlog(f"model_version: {pipeline.model_info.architecture} distilled={pipeline.is_distilled}")
        dlog(f"defaults: {defaults}")
        dlog(f"num_steps={num_steps} guidance_scale={guidance_scale} cfg_scale={cfg_scale}")
        dlog(f"state.seed={state.seed} prompt={state.prompt!r}")
        dlog(f"state.preview_width={state.preview_width} preview_height={state.preview_height}")
        dlog(f"dit dtype: {pipeline.dit_dtype}")
        dlog(f"dit device: {next(pipeline.dit.parameters()).device}")
        # Sample a weight to check its dtype
        for n, p in pipeline.dit.named_parameters():
            dlog(f"sample DiT param: {n} dtype={p.dtype} shape={list(p.shape)} device={p.device}")
            break
        # LoRA status
        if self.primary_network is not None:
            n_enabled = sum(1 for m in self.primary_network.unet_loras if m.enabled)
            n_total = len(self.primary_network.unet_loras)
            mults = {m.multiplier for m in self.primary_network.unet_loras}
            dlog(f"primary LoRA: {n_enabled}/{n_total} modules enabled, unique multipliers={mults}")

        # Round to multiples of 16 (Klein patch size).
        width = (state.preview_width // 16) * 16
        height = (state.preview_height // 16) * 16

        # TE back on GPU for encode (we offload it in ensure_pipeline for memory).
        try:
            pipeline.reload_text_encoder()
        except Exception:
            logger.exception("reload_text_encoder failed")

        # Encode prompt — use the pipeline helper (trainer-style fp8 autocast handling inside).
        prompt = state.prompt or ""
        ctx_vec, neg_ctx_vec = pipeline.encode_prompt(prompt, " ")

        # Free TE for the DiT forward.
        try:
            pipeline.unload_text_encoder()
        except Exception:
            pass
        # Phase boundary 2: reclaim TE's GPU footprint before the DiT forward
        # starts allocating activations (relevant on 16 GB cards at max swap).
        clean_memory_on_device(device)

        def _stats(t):
            tf = t.detach().float()
            return f"shape={list(t.shape)} dtype={t.dtype} device={t.device} mean={tf.mean().item():.4f} std={tf.std().item():.4f} min={tf.min().item():.4f} max={tf.max().item():.4f} norm={tf.norm().item():.2f}"

        dlog(f"ctx_vec raw: {_stats(ctx_vec)}")
        dlog(f"neg_ctx_vec raw: {_stats(neg_ctx_vec)}")

        ctx = ctx_vec.to(device=device, dtype=torch.bfloat16)
        ctx, ctx_ids = prc_txt(ctx)
        dlog(f"after prc_txt: ctx {_stats(ctx)} ctx_ids shape={list(ctx_ids.shape)} dtype={ctx_ids.dtype}")

        neg_ctx = neg_ctx_vec.to(device=device, dtype=torch.bfloat16) if neg_ctx_vec is not None else None
        neg_ctx_ids = None
        if neg_ctx is not None:
            neg_ctx, neg_ctx_ids = prc_txt(neg_ctx)

        # Seeded latent noise.
        generator = torch.Generator(device=device)
        if state.seed is not None:
            generator.manual_seed(int(state.seed))
        packed_h, packed_w = height // 16, width // 16
        latents = randn_tensor(
            (1, 128, packed_h, packed_w),
            generator=generator, device=device, dtype=torch.bfloat16,
        )
        x, x_ids = prc_img(latents)
        dlog(f"initial latent packed: {_stats(x)} x_ids shape={list(x_ids.shape)}")

        # Prepare block-swap (no-op if not enabled).
        if hasattr(pipeline.dit, "_offloader") and pipeline.dit._offloader is not None:
            pipeline.dit._offloader.set_forward_only(True)
            pipeline.dit.prepare_block_swap_before_forward()

        # flow_shift=None → Fizgig's compute_empirical_mu (gives mu ≈ 2.0 at
        # 4 steps, seq_len=1024 — matches ComfyUI's Klein shift=2.02 via the
        # generalized SNR formula).
        timesteps = get_schedule(num_steps, x.shape[1], None)
        dlog(f"timesteps ({len(timesteps)}): {[round(t, 4) for t in timesteps]}")

        pipeline.dit.eval()
        import torch as _torch

        # --- v2 Turbo: determine if activation cache is usable ---
        cache_key = (self.primary_path, self.donor_path, state.seed,
                     state.prompt, width, height)
        can_use_cache = (
            self._turbo_enabled
            and pipeline.is_distilled
            and self._act_cache is not None
            and self._act_cache_key == cache_key
            and self._changed_blocks
        )
        if can_use_cache:
            try:
                free_vram_gb = _torch.cuda.mem_get_info()[0] / (1024 ** 3)
                if free_vram_gb < 2.0:
                    dlog(f"Turbo skipped: only {free_vram_gb:.1f} GB VRAM free")
                    can_use_cache = False
            except Exception:
                pass

        resume_point = self._compute_resume_point() if can_use_cache else None
        if resume_point:
            dlog(f"Turbo resume from {resume_point[0]}_{resume_point[1]}")
        elif self._turbo_enabled and pipeline.is_distilled:
            dlog("Turbo: populating cache (first run or invalidated)")

        from fizgig.klein.model import ActivationCacheEntry

        if pipeline.is_distilled:
            guidance_vec = _torch.full((x.shape[0],), guidance_scale, device=x.device, dtype=x.dtype)
            new_cache = {}
            turbo_fallback = False

            for si, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
                t_vec = _torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=x.device)

                # Decide whether to use cached forward for this step
                use_cached = (
                    self._turbo_enabled
                    and not turbo_fallback
                    and (can_use_cache or (not can_use_cache and self._turbo_enabled))
                )

                if use_cached:
                    step_resume = resume_point if can_use_cache else None
                    step_cached = self._act_cache.get(si) if can_use_cache else None
                    new_entry = ActivationCacheEntry(
                        double_outputs=[None] * 8,
                        transition_img=None, transition_pe=None,
                        single_outputs=[None] * 24,
                    )
                    try:
                        if hasattr(pipeline.dit, 'prepare_block_swap_before_forward'):
                            pipeline.dit.prepare_block_swap_before_forward()
                        with _torch.no_grad(), _torch.autocast(device_type=x.device.type, dtype=x.dtype):
                            pred = pipeline.dit.forward_cached(
                                x=x, x_ids=x_ids, timesteps=t_vec,
                                ctx=ctx, ctx_ids=ctx_ids, guidance=guidance_vec,
                                resume_from=step_resume, cached=step_cached,
                                new_cache=new_entry,
                            )
                        new_cache[si] = new_entry
                    except Exception:
                        logger.warning("Turbo cache error at step %d; falling back to full forward", si, exc_info=True)
                        self._invalidate_activation_cache()
                        turbo_fallback = True
                        with _torch.no_grad(), _torch.autocast(device_type=x.device.type, dtype=x.dtype):
                            pred = pipeline.dit(x=x, x_ids=x_ids, timesteps=t_vec, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance_vec)
                else:
                    with _torch.no_grad(), _torch.autocast(device_type=x.device.type, dtype=x.dtype):
                        pred = pipeline.dit(x=x, x_ids=x_ids, timesteps=t_vec, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance_vec)

                dlog(f"step {si}: t_curr={t_curr:.4f} t_prev={t_prev:.4f} dt={(t_prev - t_curr):+.4f}")
                dlog(f"  pred: {_stats(pred)}")
                x = x + (t_prev - t_curr) * pred
                dlog(f"  x after step: {_stats(x)}")

            # Store cache for next preview
            if self._turbo_enabled and not turbo_fallback and new_cache:
                self._act_cache = new_cache
                self._act_cache_key = cache_key
        else:
            x = denoise_cfg(
                pipeline.dit, x, x_ids, ctx, ctx_ids, neg_ctx, neg_ctx_ids,
                timesteps=timesteps, guidance=cfg_scale,
            )
            dlog(f"after denoise_cfg: x {_stats(x)}")

        # Unpack latent → VAE decode.
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        dlog(f"unpacked latent: {_stats(x)}")
        latent = x.to(pipeline.vae.dtype)
        del x

        # Phase boundary 3: release DiT activation scratch before VAE claims GPU.
        clean_memory_on_device(device)

        pipeline.vae.to(device)
        pipeline.vae.eval()
        with torch.no_grad():
            pixels = pipeline.vae.decode(latent)
        dlog(f"vae decoded: {_stats(pixels)}")
        del latent
        pixels = pixels.to(torch.float32).cpu()
        pixels = (pixels / 2 + 0.5).clamp(0, 1)
        pipeline.vae.to("cpu")
        # Phase boundary 4: release VAE decode scratch before handing PIL back.
        clean_memory_on_device(device)

        img_np = (pixels[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        dlog(f"img_np stats: shape={img_np.shape} dtype={img_np.dtype} mean={img_np.mean():.2f} std={img_np.std():.2f}")
        img = Image.fromarray(img_np)

        # Dump log to a file users can share.
        try:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            log_path = os.path.join(repo_root, "repair_studio_last_preview.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(log_lines))
            logger.info("Wrote diagnostic log to %s", log_path)
        except Exception:
            logger.exception("Failed to write diagnostic log")

        # v2 hook: clear after preview so next mark_blocks_changed accumulates fresh.
        self._changed_blocks.clear()
        return img

    def generate_baseline(self, state: SliderState) -> Image.Image:
        """Baseline = primary at default 1.0 / all enabled, donor fully off.

        Cached on (primary_path, seed, prompt, w, h) — slider tweaks don't
        invalidate it. Cache cleared on primary swap or reset.
        """
        key = (self.primary_path, state.seed, state.prompt, state.preview_width, state.preview_height)
        if self._baseline_cache_key == key and self._baseline_cache_image is not None:
            return self._baseline_cache_image

        baseline_state = SliderState.default_klein9b()
        baseline_state.seed = state.seed
        baseline_state.prompt = state.prompt
        baseline_state.preview_width = state.preview_width
        baseline_state.preview_height = state.preview_height

        img = self.generate_preview(baseline_state)
        self._baseline_cache_key = key
        self._baseline_cache_image = img
        return img

    def _invalidate_baseline_cache(self) -> None:
        self._baseline_cache_key = None
        self._baseline_cache_image = None
