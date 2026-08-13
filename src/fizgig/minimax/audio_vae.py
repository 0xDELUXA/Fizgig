"""MiniMax H3 audio VAE — encoder half only (waveform -> 32-ch latents at 40 Hz).

H3 denoises audio and video jointly, and Fizgig's DiT side has always packed the audio rows —
but as SILENCE, because nothing could turn a waveform into latents. This is that missing piece:
with it, a training clip's real sound becomes a real target.

Ported from the ComfyUI reference (`comfy/ldm/minimax/audio_vae.py`, DAC-lineage encoder; the
BigVGAN decoder there is NOT ported — see below). Same treatment as the video VAE port next door:
comfy's `ops` become plain `nn` modules, and the checkpoint loads with `strict=False` so the
decoder half of the file is simply ignored.

Shapes and rates, all of which the DiT already assumes:

  * input   stereo waveform [B, 2, L], 32 kHz, in [-1, 1]
  * hop     800 samples -> 32000/800 = **40 latents per second**, which is exactly the
            `AUDIO_LATENTS_PER_SECOND` model.py has always been written against
  * output  [B, 32, 2, T] normalized latents; 32 is `audio_latents_dim`

Each channel is encoded independently — the encoder is mono and the reference reshapes stereo to
`b*2` before it. Length is right-padded with zeros to a multiple of the hop.

WHY NO DECODER: previews with sound would need BigVGAN, roughly the other half of the reference
file. Judging an H3 checkpoint in ComfyUI is already the doctrine and ComfyUI has the decoder, so
audio training can be validated without it. Deferred, not rejected.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 32000
HOP_LENGTH = 800                 # prod of the encoder strides (2*4*4*5*5)
LATENTS_PER_SECOND = SAMPLE_RATE // HOP_LENGTH      # 40 — matches model.AUDIO_LATENTS_PER_SECOND


def snake(x, alpha, beta):
    """x + 1/beta * sin^2(alpha * x). Not in-place: the reference can mutate its own temporaries,
    this runs under autograd where that would break the backward."""
    t = torch.sin(alpha * x)
    return x + (t * t) * (beta + 1e-9).reciprocal()


class Snake1d(nn.Module):
    """Snake with one alpha per channel, alpha serving as beta too (the encoder's variant)."""

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        a = self.alpha.to(x)
        return snake(x, a, a)


class ResidualUnit(nn.Module):
    def __init__(self, dim=16, dilation=1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return y + x


class EncoderBlock(nn.Module):
    def __init__(self, dim=16, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            nn.Conv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride,
                      padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """DAC encoder: [B, 1, L] waveform -> [B, d_latent, L/800]."""

    def __init__(self, d_model=64, strides=(2, 4, 4, 5, 5), d_latent=2048):
        super().__init__()
        block = [nn.Conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            block += [EncoderBlock(d_model, stride=stride)]
        block += [Snake1d(d_model), nn.Conv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class GeGluMlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.norm(x)
        return self.w2(self.act(self.w0(x)) * self.w1(x))


class CausalAttention(nn.Module):
    """Causal attention that also POOLS 2048 -> 32.

    Two easy things to get wrong, both mirrored from the reference:
      * qkv has no bias parameter of its own; q and v biases are separate tensors and k's bias is
        a registered ZERO buffer, concatenated at call time. Dropping the zeros shifts k.
      * the head axis is averaged (not concatenated) and then adaptive-avg-pooled down to the
        latent width, which is how 2048 becomes 32.
    """

    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.head_dim = in_dim // num_heads
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(in_dim))
        self.v_bias = nn.Parameter(torch.zeros(in_dim))
        self.register_buffer("zero_k_bias", torch.zeros(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        B, N, _ = x.shape
        bias = torch.cat((self.q_bias, self.zero_k_bias.to(self.q_bias), self.v_bias)).to(x)
        qkv = F.linear(x, self.qkv.weight.to(x), bias)
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = F.adaptive_avg_pool1d(torch.mean(x, dim=1), self.out_dim)
        return self.proj(x)


class AttnProjection(nn.Module):
    """The posterior head: [B, T, 2048] -> [B, T, 32]."""

    def __init__(self, in_dim, out_dim, num_heads, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = GeGluMlp(in_features=out_dim, hidden_features=int(out_dim * mlp_ratio))

    def forward(self, x):
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def pack_audio(latent: torch.Tensor) -> torch.Tensor:
    """[B, 32, 2, T] -> [2*T, 32] rows, CHANNEL-MAJOR (ch0 t0..T-1, then ch1 t0..T-1).

    The ordering is a checkpoint contract, not a choice — it is what the frozen base was trained
    on, and it is `pack_audio` in the reference DiT verbatim. Time-major would run silently and
    train against scrambled targets.
    """
    _, c, ch, t = latent.shape
    return latent[0].permute(1, 2, 0).reshape(ch * t, c)


class MiniMaxH3AudioVAEEncoder(nn.Module):
    """Encode-only. The official `minimax_h3_audio_vae_fp32.safetensors` loads with
    `strict=False`; the decoder / logs_proj / dec_in_proj keys are simply unused."""

    def __init__(self, encoder_dim=64, encoder_rates=(2, 4, 4, 5, 5),
                 latent_dim=2048, vae_latent_channels=32):
        super().__init__()
        self.sample_rate = SAMPLE_RATE
        self.hop_length = int(math.prod(encoder_rates))          # 800
        self.latents_per_second = self.sample_rate // self.hop_length
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)
        self.pre_block = AttnProjection(latent_dim, vae_latent_channels, num_heads=8)
        self.mean_proj = nn.Conv1d(vae_latent_channels, vae_latent_channels, 1)
        self.register_buffer("latents_mean", torch.zeros(vae_latent_channels))
        self.register_buffer("latents_std", torch.ones(vae_latent_channels))

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Stereo [B, 2, L] at 32 kHz in [-1, 1] -> normalized latents [B, 32, 2, T].

        The posterior MEAN is used directly, with no sampling — same choice as the video VAE
        path, where a frozen draw is strictly worse than the mean for a cached target.
        """
        b, s, length = waveform.shape
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        if right_pad:
            waveform = F.pad(waveform, (0, right_pad))
        x = self.encoder(waveform.reshape(b * s, 1, -1))
        x = self.pre_block(x.transpose(1, 2)).transpose(1, 2)
        z = self.mean_proj(x)
        mean = self.latents_mean.view(1, -1, 1).to(z)
        std = self.latents_std.view(1, -1, 1).to(z)
        z = (z - mean) / std
        return z.reshape(b, s, z.shape[1], z.shape[2]).permute(0, 2, 1, 3)


def load_minimax_h3_audio_vae(path: str, device="cuda", dtype=torch.float32
                              ) -> MiniMaxH3AudioVAEEncoder:
    """Build the encoder and load the official checkpoint, ignoring the decoder half.

    Read through MemoryEfficientSafeOpen for the same reason every other loader here does: the
    official safe_open mmap path hard-crashes on Windows for large files.
    """
    from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen

    model = MiniMaxH3AudioVAEEncoder()
    wanted = set(model.state_dict().keys())
    sd = {}
    with MemoryEfficientSafeOpen(path) as f:
        for k in f.keys():
            if k in wanted:
                sd[k] = f.get_tensor(k)
    missing, _unexpected = model.load_state_dict(sd, strict=False)
    # zero_k_bias is a constant this port registers itself; anything else missing means the file
    # is not the encoder we think it is, and a silently random encoder would poison every cached
    # audio target without raising.
    real_missing = [m for m in missing if not m.endswith("zero_k_bias")]
    if real_missing:
        raise ValueError(
            f"{path} is missing {len(real_missing)} audio-encoder weight(s), e.g. "
            f"{real_missing[:3]} — expected the MiniMax H3 audio VAE "
            f"(minimax_h3_audio_vae_fp32.safetensors).")
    return model.to(device=device, dtype=dtype).eval()
