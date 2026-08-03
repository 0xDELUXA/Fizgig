"""MiniMax H3 video VAE — ENCODE PATH ONLY (image -> 24-channel latent).

Pure-PyTorch port of the encoder half of ComfyUI's comfy/ldm/minimax/vae.py. Image-only
training needs to turn a still image into the DiT's 24-channel latent exactly once (caching),
so only the 3D-causal-CNN encoder + quant_conv + latent normalization are ported. The ViT3D
decoder, spatial tiling and temporal chunking are omitted — no sampling/decode in scope.

Weight names match the checkpoint's `encoder.*` / `quant_conv.*` / `latents_mean/std`, so the
official minimax_h3_video_vae_fp16.safetensors loads with strict=False (decoder/post_quant keys
ignored). A single image is one video frame (T=1): 16x spatial downscale, T stays 1.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608886, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.4498890042304993, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235595, 3.0496184825897216, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811524,
]


class CausalConv3d(nn.Conv3d):
    """Reflect spatial padding, causal (front-only, zero) temporal padding."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.causal_padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)

    def forward(self, x):
        cp = self.causal_padding
        if sum(cp) == 0:
            return super().forward(x)
        # spatial reflect (H, W), then temporal causal front-zeros — unifies the reference's
        # single-frame and multi-frame paths (front-pad by 2*cp[0] zeros is numerically the
        # single-frame "causal_zero" optimization).
        x = F.pad(x, (cp[2], cp[2], cp[1], cp[1], 0, 0), mode="reflect")
        x = F.pad(x, (0, 0, 0, 0, cp[0] * 2, 0), mode="constant")
        return super().forward(x)


class TemporalIsolatedGroupNorm(nn.GroupNorm):
    """GroupNorm with per-frame statistics (time folded into batch)."""
    def forward(self, x):
        if x.dim() == 5:
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, 1, h, w)
            x = super().forward(x)
            return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return super().forward(x)


def group_norm_3d(num_channels):
    return TemporalIsolatedGroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_stride=1, space_stride=2):
        super().__init__()
        self.space_stride = space_stride
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=(1, 0, 0),
                                 stride=(time_stride, space_stride, space_stride))

    def forward(self, x):
        if self.space_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = group_norm_3d(in_channels)
        self.norm2 = group_norm_3d(out_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return h + x


class EncoderFCN3D(nn.Module):
    def __init__(self, ch, ch_mult, space_down, time_down, num_res_blocks, in_channels, z_channels, double_z=True):
        super().__init__()
        self.num_levels = len(ch_mult)
        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * self.num_levels
        self.num_res_blocks = num_res_blocks

        block_mid = [ch * ch_mult[i] for i in range(self.num_levels)]
        block_in = [block_mid[0]] + block_mid[:-1]
        block_out = block_mid

        self.conv_in = CausalConv3d(in_channels, block_in[0], kernel_size=3, padding=1)
        self.down = nn.ModuleList()
        for i_level in range(self.num_levels):
            down = nn.Module()
            down.block = nn.ModuleList()
            for i in range(self.num_res_blocks[i_level]):
                down.block.append(ResnetBlock3D(
                    in_channels=block_in[i_level] if i == 0 else block_mid[i_level],
                    out_channels=block_mid[i_level]))
            if space_down[i_level] * time_down[i_level] > 1:
                down.downsample = Downsample3D(block_mid[i_level], block_out[i_level],
                                               time_stride=time_down[i_level], space_stride=space_down[i_level])
            self.down.append(down)
        self.norm_out = group_norm_3d(block_out[-1])
        self.conv_out = CausalConv3d(block_out[-1], 2 * z_channels if double_z else z_channels,
                                     kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_levels):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](h)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


class MiniMaxH3VideoVAEEncoder(nn.Module):
    """Encode-only. Load the full checkpoint with strict=False (decoder keys ignored)."""

    def __init__(self, in_channels=3, ch=128, embed_dim=24, z_channels=24,
                 ch_mult=(1, 2, 2, 4, 4, 8), num_res_blocks=2,
                 space_down=(2, 2, 2, 2, 1, 1), time_down=(1, 2, 2, 1, 1, 1)):
        super().__init__()
        self.vae_ratio = int(math.prod(space_down))        # 16
        self.vae_ratio_t = int(math.prod(time_down))       # 4
        self.encoder = EncoderFCN3D(ch=ch, ch_mult=list(ch_mult), space_down=list(space_down),
                                    time_down=list(time_down), num_res_blocks=num_res_blocks,
                                    in_channels=in_channels, z_channels=z_channels, double_z=True)
        self.quant_conv = nn.Conv3d(z_channels * 2, 2 * embed_dim, 1)
        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN[:embed_dim]))
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD[:embed_dim]))
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)

    @torch.no_grad()
    def encode(self, x):
        """x: [B,3,H,W] or [B,3,T,H,W] in [-1,1] -> normalized latent [B,24,1,H/16,W/16] (image: T=1)."""
        if x.ndim == 4:
            x = x.unsqueeze(2)
        x = (x + 1.0) * 0.5
        x = (x - self.pixel_mean.to(x)) / self.pixel_std.to(x)
        if x.shape[2] != 1:
            raise NotImplementedError("MiniMax H3 image training encodes a single frame (T=1)")
        moments = self.quant_conv(self.encoder(x))
        moments = moments[:, :, -1:, :, :]
        mean = torch.chunk(moments.float(), 2, dim=1)[0]
        lm = self.latents_mean.view(1, -1, 1, 1, 1).to(mean)
        ls = self.latents_std.view(1, -1, 1, 1, 1).to(mean)
        return (mean - lm) / ls
