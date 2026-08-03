"""MiniMax H3 — image sampling for in-training previews.

Renders ONE still per prompt. H3's temporal grid is FRAME_PER_TOKEN = (1, 4, 4, 4, 4), so a
single latent frame decodes to exactly one pixel frame — which is also what the encoder produces
for a still, and what training uses. So a preview needs no audio rows, no multi-frame RoPE and no
packed-layout work: it is the training forward run backwards.

A preview is therefore a *training-distribution* render, not a byte-faithful ComfyUI one (ComfyUI
always packs audio rows and >= 2 latent frames). It answers "is this LoRA learning?"; final quality
judgement belongs in ComfyUI. Same honest caveat Klein's samples carry.

Flow / sign convention (this is the easy thing to get backwards):
  the DiT head returns `video_out = x0 - noise` (the reference NEGATES it to hand a sampler its
  velocity, then uses denoised = x - v*sigma). Integrating our RAW head output from sigma=1 down
  to 0 is therefore:

      x += (sigma_curr - sigma_next) * model_out

  Check: from sigma=1 (x = noise), one step to sigma=0 gives noise + (x0 - noise) = x0.
  Note this is the opposite arrangement to Krea 2's `img += (t_prev - t_curr) * v`, whose t runs
  1 -> 0 rather than being a noise level.
"""

import gc
import logging

import torch

logger = logging.getLogger(__name__)

# Per-voxel linear 24ch-latent -> RGB approximation (ComfyUI's MiniMaxH3Video.latent_rgb_factors).
# A free, instant, very rough preview: no decoder, no VRAM. Good enough to confirm a sampler is
# producing structure; NOT good enough for likeness scoring — that needs the real VAE decoder.
LATENT_RGB_FACTORS = [
    [-0.018555, 0.024344, -0.017536], [0.150164, 0.137244, 0.129221],
    [0.027367, -0.050369, -0.208606], [-0.000793, -0.164622, -0.323161],
    [-0.048556, 0.013970, -0.074286], [0.011740, 0.014172, -0.006906],
    [0.061517, 0.061212, 0.110025], [0.035321, 0.086879, 0.110059],
    [-0.017426, 0.002997, 0.035356], [0.531539, 0.548819, 0.624404],
    [-0.024968, -0.040234, -0.034302], [-0.032549, -0.029096, -0.017221],
    [0.022609, 0.020286, 0.050661], [-0.084001, -0.038131, -0.020805],
    [-0.018830, 0.010412, 0.061120], [0.020777, 0.011196, -0.030994],
    [-0.008390, -0.012201, -0.025687], [-0.013281, -0.002924, 0.006331],
    [0.000260, 0.001833, -0.011038], [0.105471, 0.100482, 0.132106],
    [0.016529, 0.015213, 0.009999], [-0.014015, -0.017438, -0.019134],
    [-0.033787, -0.009984, -0.019725], [0.004224, 0.017284, 0.027196],
]
LATENT_RGB_BIAS = [0.057426, -0.022078, -0.071449]


def sample_schedule(steps: int, shift: float = 12.0):
    """Descending sigmas 1 -> 0 on H3's shifted grid: sigma = shift*u / (1 + (shift-1)*u).

    shift 12.0 is the video schedule the reference sampler and every shipped H3 workflow use
    (supported_models sampling_settings). Returns steps+1 values, last exactly 0.0."""
    out = []
    for i in range(steps):
        u = 1.0 - i / steps
        out.append((shift * u) / (1.0 + (shift - 1.0) * u))
    out.append(0.0)
    return out


def latent_to_rgb(latent: torch.Tensor):
    """[1, 24, 1, H, W] latent -> uint8 HWC array, via the linear RGB approximation.

    Upscales nothing: the result is one pixel per latent cell (1/16 scale), so it is a thumbnail
    of a thumbnail. Purely a sanity view until the real decoder lands."""
    import numpy as np
    f = torch.tensor(LATENT_RGB_FACTORS, dtype=torch.float32, device=latent.device)
    b = torch.tensor(LATENT_RGB_BIAS, dtype=torch.float32, device=latent.device)
    z = latent.detach().float()[0, :, 0]                  # [24, H, W]
    rgb = torch.einsum("chw,cr->rhw", z, f) + b[:, None, None]
    rgb = rgb.clamp(0, 1).mul(255).round().byte().cpu().numpy()
    return np.transpose(rgb, (1, 2, 0))                   # HWC


@torch.no_grad()
def sample_image(model, text_embeds, *, width=512, height=512, steps=8, cfg_scale=1.0,
                 uncond_embeds=None, seed=0, shift=12.0, device="cuda",
                 dtype=torch.bfloat16, latent_channels=24, spatial=16):
    """Denoise one image and return its LATENT [1, 24, 1, H/16, W/16].

    Decoding is the caller's business (real VAE decoder, or latent_to_rgb for a rough look) so
    this stays testable without a 4.8 GB decoder resident.

    cfg_scale <= 1.0 runs a single forward per step (what every shipped H3 workflow does); above
    that costs a second forward, because the DiT is hard-locked to batch size 1.
    """
    lat_h, lat_w = height // spatial, width // spatial
    # The DiT patchifies 2x2, so the latent grid must be even (compute_loss crops for the same
    # reason on the training side).
    lat_h, lat_w = (lat_h // 2) * 2, (lat_w // 2) * 2
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn(1, latent_channels, 1, lat_h, lat_w, generator=gen, dtype=torch.float32).to(device)

    use_cfg = cfg_scale > 1.0 and uncond_embeds is not None
    sigmas = sample_schedule(steps, shift=shift)
    for i in range(steps):
        s_curr, s_next = sigmas[i], sigmas[i + 1]
        t = torch.tensor([1.0 - s_curr], device=device)     # the DiT is conditioned on cleanness
        out = model(x.to(dtype), t, text_embeds).float()
        if use_cfg:
            out_u = model(x.to(dtype), t, uncond_embeds).float()
            out = out_u + cfg_scale * (out - out_u)
        x = x + (s_curr - s_next) * out                     # see the sign note in the docstring
    return x


def encode_sample_prompts(te_path, prompts, *, device="cuda", quantize=True):
    """Pre-encode preview prompts, then free the encoder.

    Always call this BEFORE the DiT loads: the Qwen3-VL-32B text encoder is ~14 GB NF4 and must
    never be resident alongside the ~17 GB DiT. Returns CPU tensors."""
    from fizgig.minimax.embedder import load_minimax_h3_te
    te = load_minimax_h3_te(te_path, device=device, compute_dtype=torch.bfloat16, quantize=quantize)
    out = [te.encode(p).cpu() for p in prompts]
    del te
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out
