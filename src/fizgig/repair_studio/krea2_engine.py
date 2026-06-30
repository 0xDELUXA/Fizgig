"""Repair-Studio engine for Krea 2 — the parallel of `engine.RepairEngine`.

Klein's RepairEngine inlines a 270-line Klein denoise loop that doesn't transfer; Krea 2's
`sampling.sample` is a complete sampler, so the Krea 2 preview is simpler: apply the slider
state to the LoRA networks (the model-agnostic `set_module_*_by_pattern` API) and call
`sampling.sample`. Previews always render on the fp8 Turbo (8-step, CFG-free). The per-block
slider config is the shared `SliderState`; only the block ids/regex are Krea 2-specific
(`krea2_blocks`).

Public surface mirrors RepairEngine so the Repair Studio UI can drive either: `ensure_pipeline`,
`load_primary`, `load_donor` / `unload_donor`, `apply_state`, `generate_preview`, `reset`, and
the `primary_network` / `donor_network` / `*_block_ids` / `*_path` / `primary_hash` attributes,
plus a `pipeline` holder exposing `is_loaded`.
"""

import gc
import logging
import threading
from typing import Optional, Set

import torch
from PIL import Image

from fizgig.repair_studio.krea2_blocks import block_regex_krea2, extract_block_ids_krea2

logger = logging.getLogger(__name__)


class _Loaded:
    """Tiny stand-in for KleinInferencePipeline.is_loaded so the shared UI checks
    `engine.pipeline.is_loaded` uniformly across both engines."""

    def __init__(self):
        self.is_loaded = True


def _apply_lora(target, sd, multiplier, device, dtype):
    """Normalize foreign formats, build the network, apply it live, load the weights. Returns
    the network. Mirrors the verified Context-LoRA / preview path (ensure_kohya -> apply_to ->
    load_state_dict — create_network_from_weights only builds STRUCTURE, the values must be
    loaded or lora_up stays 0)."""
    from fizgig.networks.lora import create_network_from_weights, ensure_kohya_lora_state_dict
    sd = ensure_kohya_lora_state_dict(sd)
    net = create_network_from_weights(None, float(multiplier), sd, None, target, for_inference=True)
    net.apply_to(text_encoders=None, unet=target, apply_text_encoder=False, apply_unet=True)
    net.load_state_dict(sd, strict=False)
    net.to(device=device, dtype=dtype).eval()
    return net


class Krea2RepairEngine:
    def __init__(self):
        self.pipeline: Optional[_Loaded] = None
        self.turbo = None          # fp8 Turbo DiT
        self.ae = None             # Qwen-Image VAE (kept on CPU; sampling moves it for decode)
        self.te_path: Optional[str] = None
        self.device = "cuda"
        self.dtype = torch.bfloat16

        self.primary_network = None
        self.donor_network = None
        self.primary_path: Optional[str] = None
        self.donor_path: Optional[str] = None
        self.primary_block_ids: Set[str] = set()
        self.donor_block_ids: Set[str] = set()
        self.primary_hash: Optional[str] = None

        # Encoded-prompt cache: prompt -> (txt, txtmask) on CPU. Slider tweaks don't change the
        # prompt, so the 8 GB TE only loads when the prompt actually changes.
        self._prompt_cache_key: Optional[str] = None
        self._prompt_cache = None
        # Cooperative cancellation: set to abort an in-flight render so a new edit restarts it
        # immediately instead of queueing behind the full 8-step pass.
        self._cancel_event = threading.Event()
        # Baseline render cache (LoRA at original strengths; slider tweaks don't invalidate it).
        self._baseline_cache_key = None
        self._baseline_cache_image: Optional[Image.Image] = None

        # Turbo Preview — per-step activation cache (forward_cached). Tweaking a late block
        # skips the earlier blocks. Cache key: (primary, donor, seed, prompt, w, h). The resume
        # point is computed by DIFFING the render's state against `_act_cache_state` (the state
        # the cache was built from) — not a mutable changed-set — so rapid edits during an
        # in-flight render are never lost.
        self._turbo_enabled = True
        self._act_cache = None          # {step_idx: KreaActivationCacheEntry}
        self._act_cache_key = None
        self._act_cache_state = None    # SliderState the current cache reflects

    # ----- pipeline + LoRA loading -------------------------------------------
    def ensure_pipeline(self, turbo_path: str, vae_path: str, text_encoder_path: str,
                        device: str = "cuda", model_kind: str = "turbo", **_ignored) -> None:
        """Load the preview DiT + VAE once (TE loads on demand per prompt-encode).

        model_kind: 'turbo' (fp8, 8-step, CFG-free — the default) or 'raw' (the undistilled
        base — slower, more steps). The DiT loader auto-detects pre-quant fp8 either way."""
        if self.pipeline is not None and self.pipeline.is_loaded:
            return
        from fizgig.krea2.utils import load_krea2_dit
        from fizgig.krea2.vae_loader import load_vae
        self.device = device
        self.te_path = text_encoder_path
        self.model_kind = model_kind
        self._default_steps = 20 if model_kind == "raw" else 8
        self.turbo = load_krea2_dit(turbo_path, device=device, dtype=self.dtype)  # prequant fp8 auto-detected
        self.ae = load_vae(vae_path, input_channels=3, device="cpu", disable_mmap=True)
        self.turbo.eval()
        self.pipeline = _Loaded()
        logger.info("Krea2 Repair engine ready (%s=%s)", model_kind, turbo_path)

    def load_primary(self, path: str) -> None:
        if self.pipeline is None or not self.pipeline.is_loaded:
            raise RuntimeError("Pipeline not loaded; call ensure_pipeline() first.")
        if self.primary_network is not None:
            raise RuntimeError("Primary already loaded — call reset() to swap.")
        from safetensors.torch import load_file
        self.primary_network = _apply_lora(self.turbo, load_file(path), 1.0, self.device, self.dtype)
        self.primary_path = path
        self.primary_block_ids = extract_block_ids_krea2(self.primary_network)
        self._invalidate_baseline_cache()
        try:
            from fizgig.profiler.visualize import compute_lora_hash
            self.primary_hash = compute_lora_hash(path)
        except Exception:
            self.primary_hash = None
        logger.info("Krea2 primary loaded: %s (%d blocks)", path, len(self.primary_block_ids))

    def load_donor(self, path: str) -> None:
        if self.primary_network is None:
            raise RuntimeError("Load primary LoRA before donor.")
        if self.donor_network is not None:
            raise RuntimeError("Donor already loaded — unload_donor() or reset() first.")
        from safetensors.torch import load_file
        net = _apply_lora(self.turbo, load_file(path), 1.0, self.device, self.dtype)
        net.set_enabled(False)  # donor blocks are opt-in per-slider
        self.donor_network = net
        self.donor_path = path
        self.donor_block_ids = extract_block_ids_krea2(net)
        logger.info("Krea2 donor loaded: %s (%d blocks)", path, len(self.donor_block_ids))

    def unload_donor(self) -> None:
        if self.donor_network is not None:
            self.donor_network.set_enabled(False)
            self.donor_network = None
            self.donor_path = None
            self.donor_block_ids = set()

    # ----- slider state ------------------------------------------------------
    def apply_state(self, state) -> None:
        """Push the per-block slider config into the live networks (regex-based, no reload)."""
        if self.primary_network is None:
            return
        for bid, bs in state.blocks.items():
            try:
                pat = block_regex_krea2(bid)
            except ValueError:
                continue
            self.primary_network.set_module_enabled_by_pattern(pat, bool(bs.primary_enabled))
            self.primary_network.set_module_multiplier_by_pattern(pat, float(bs.primary_strength))
            if self.donor_network is not None:
                self.donor_network.set_module_enabled_by_pattern(pat, bool(bs.donor_enabled))
                self.donor_network.set_module_multiplier_by_pattern(pat, float(bs.donor_strength))

    # ----- cancellation ------------------------------------------------------
    def request_cancel(self) -> None:
        """Signal the in-flight render to abort at the next step boundary."""
        self._cancel_event.set()

    def clear_cancel(self) -> None:
        self._cancel_event.clear()

    # ----- preview -----------------------------------------------------------
    def _encode_prompt(self, prompt: str):
        """Encode the prompt once (load TE -> encode -> free), cached on the prompt string."""
        if self._prompt_cache_key == prompt and self._prompt_cache is not None:
            return self._prompt_cache
        from fizgig.krea2.utils import load_krea2_text_encoder
        from fizgig.krea2 import sampling
        enc = load_krea2_text_encoder(self.te_path, dtype=self.dtype, device=self.device)
        txt, txtmask, _, _ = sampling.encode_prompts(enc, [prompt], cfg=False)
        del enc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._prompt_cache_key = prompt
        self._prompt_cache = (txt.cpu(), txtmask.cpu())
        return self._prompt_cache

    def generate_preview(self, state, *, seed: Optional[int] = None,
                         prompt: Optional[str] = None, width: Optional[int] = None,
                         height: Optional[int] = None, steps: Optional[int] = None) -> Image.Image:
        """Apply the slider state and render one preview with the live LoRA(s). Always a full
        forward — the per-step activation cache (forward_cached) is NOT used here.

        Why: across a multi-step denoise, img is chained (img_{t+1} = img_t + dt*model(img_t)).
        A block tweak changes step 0's output, which changes step 1's input img, which makes the
        cached prefix of EVERY later step stale. Reusing it (Klein's "approximate" cache) is
        visually fine at Klein's 4 steps but compounds badly over krea's 8 — the symptom was
        tweaks only partially registering. forward_cached stays on the model (correct within a
        single pass, and a future multi-step-aware cache could use it), but the live preview does
        the correct full forward, which on the 8-step Turbo is already fast (~7 s)."""
        self.apply_state(state)
        prompt = prompt if prompt is not None else state.prompt
        seed = seed if seed is not None else state.seed
        width = width or state.preview_width
        height = height or state.preview_height
        steps = steps or getattr(self, "_default_steps", 8)
        txt, txtmask = self._encode_prompt(prompt)
        txt = txt.to(self.device)
        txtmask = txtmask.to(self.device)

        from fizgig.krea2 import sampling
        with torch.no_grad():
            imgs = sampling.sample(self.turbo, self.ae, txt, txtmask, untxt=None, untxtmask=None,
                                   device=self.device, dtype=self.dtype, width=width, height=height,
                                   steps=steps, cfg_scale=1.0, mu=1.15, seed=seed,
                                   should_abort=self._cancel_event.is_set)
        return imgs[0]

    def _sample_cached(self, txt, txtmask, *, seed, width, height, steps, resume_from):
        """Denoise loop mirroring sampling.sample but via forward_cached, threading a per-step
        activation cache. Turbo preview is CFG-free (cfg_scale=1.0), so only the cond path runs."""
        from einops import rearrange
        from fizgig.krea2.sampling import prepare, timesteps, roundup
        from fizgig.krea2.model import KreaActivationCacheEntry
        model, ae, device, dtype = self.turbo, self.ae, self.device, self.dtype
        patch = model.config.patch
        compression = 2 ** len(ae.temperal_downsample)
        channels = ae.z_dim
        align = compression * patch
        width, height = roundup(width, align, "width"), roundup(height, align, "height")
        noise = torch.randn(1, channels, height // compression, width // compression,
                            device=device, dtype=dtype,
                            generator=torch.Generator(device=device).manual_seed(seed))
        img, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)
        x1 = (256 // align) ** 2
        x2 = (1280 // align) ** 2
        ts = timesteps(img.shape[1], steps, x1, x2, y1=0.5, y2=1.15, mu=1.15)
        new_cache = {}
        prev = self._act_cache or {}
        use_resume = resume_from is not None and bool(prev)
        with torch.autocast(device_type=torch.device(device).type, dtype=dtype):
            for si, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
                t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
                entry = KreaActivationCacheEntry(block_inputs=[])
                step_cached = prev.get(si) if use_resume else None
                step_resume = resume_from if step_cached is not None else None
                v = model.forward_cached(img=img, context=txt, t=t, pos=pos, mask=mask,
                                         resume_from=step_resume, cached=step_cached, new_cache=entry)
                new_cache[si] = entry
                img = img + (tprev - tcurr) * v
        img = rearrange(img, "b (h w) (c ph pw) -> b c 1 (h ph) (w pw)",
                        ph=patch, pw=patch, h=height // align, w=width // align)
        ae = ae.to(img.device)
        pixels = ae.decode_to_pixels(img.to(torch.bfloat16))
        self.ae = ae.to("cpu")
        pixels = rearrange(pixels * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
        return Image.fromarray(pixels[0]), new_cache

    def generate_baseline(self, state) -> Image.Image:
        """Baseline = primary at default 1.0 / all enabled, donor off. Cached on
        (primary_path, seed, prompt, w, h) — slider tweaks don't invalidate it."""
        from fizgig.repair_studio.state import SliderState
        key = (self.primary_path, state.seed, state.prompt,
               state.preview_width, state.preview_height)
        if self._baseline_cache_key == key and self._baseline_cache_image is not None:
            return self._baseline_cache_image
        base = SliderState.default_krea2()
        base.seed = state.seed
        base.prompt = state.prompt
        base.preview_width = state.preview_width
        base.preview_height = state.preview_height
        img = self.generate_preview(base)
        self._baseline_cache_key = key
        self._baseline_cache_image = img
        return img

    def _invalidate_baseline_cache(self) -> None:
        self._baseline_cache_key = None
        self._baseline_cache_image = None

    def _invalidate_activation_cache(self) -> None:
        self._act_cache = None
        self._act_cache_key = None
        self._act_cache_state = None

    def mark_blocks_changed(self, blocks) -> None:
        # The GUI calls this on every slider change, but the resume point is derived from a
        # state diff at render time (race-free), so this is just a no-op compatibility shim.
        pass

    @staticmethod
    def _block_index(block_id):
        return int(block_id.split("_")[1]) if str(block_id).startswith("block_") else None

    def _resume_from_diff(self, state):
        """Earliest main-block index whose primary/donor differs from the cached state, or None
        (full recompute) if a non-main block (txtfusion) changed or there's no cached state.
        Diffing the FULL state against what the cache holds guarantees no edit is ever missed,
        even ones made while a previous render was in flight."""
        if self._act_cache_state is None:
            return None
        changed = state.diff_blocks(self._act_cache_state)
        if not changed:
            return None
        idxs = [self._block_index(b) for b in changed]
        if any(i is None for i in idxs):  # a txtfusion / non-main block changed -> full pass
            return None
        return min(idxs)

    # ----- teardown ----------------------------------------------------------
    def reset(self) -> None:
        """Full unload — drop networks (break forward-hook ref cycles) then the Turbo + VAE."""
        for net in (self.primary_network, self.donor_network):
            if net is not None:
                try:
                    for lora in net.unet_loras:
                        lora.org_forward = None
                    net.unet_loras.clear()
                except Exception:
                    pass
        self.primary_network = None
        self.donor_network = None
        self.turbo = None
        self.ae = None
        self.pipeline = None
        self.primary_path = None
        self.donor_path = None
        self.primary_block_ids = set()
        self.donor_block_ids = set()
        self.primary_hash = None
        self._prompt_cache_key = None
        self._prompt_cache = None
        self._baseline_cache_key = None
        self._baseline_cache_image = None
        self._act_cache = None
        self._act_cache_key = None
        self._act_cache_state = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
