"""LoRA Profiler: differential forward passes to map LoRA activity per block per timestep.

Runs the model with and without LoRA at different noise levels, compares activation
differences, and produces a per-block per-timestep activity map.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from fizgig.klein.inference import KleinInferencePipeline
from fizgig.klein.position import prc_img, prc_txt

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Result of a LoRA profiling run."""
    block_names: list[str]
    timestep_bins: list[tuple[float, float]]  # (low, high) per bin
    magnitudes: np.ndarray  # shape (num_blocks, num_bins)
    metadata: dict = field(default_factory=dict)

    @property
    def num_blocks(self) -> int:
        return len(self.block_names)

    @property
    def num_bins(self) -> int:
        return len(self.timestep_bins)

    def get_top_blocks(self, n: int = 10) -> list[tuple[str, float]]:
        """Return the top N blocks by total activity across all timesteps."""
        totals = self.magnitudes.sum(axis=1)
        indices = np.argsort(totals)[::-1][:n]
        return [(self.block_names[i], totals[i]) for i in indices]

    def get_block_activity(self, block_name: str) -> Optional[np.ndarray]:
        """Get per-timestep activity for a specific block."""
        if block_name in self.block_names:
            idx = self.block_names.index(block_name)
            return self.magnitudes[idx]
        return None


class LoRAProfiler:
    """Profile a LoRA by running differential forward passes.

    Usage:
        pipeline = KleinInferencePipeline()
        pipeline.load_models(...)

        profiler = LoRAProfiler(pipeline)
        result = profiler.profile("path/to/lora.safetensors")

        from fizgig.profiler.visualize import plot_profile_heatmap
        plot_profile_heatmap(result, "profile.png")
    """

    def __init__(self, pipeline: KleinInferencePipeline):
        self.pipeline = pipeline

    def profile(
        self,
        lora_path: str,
        num_samples: int = 8,
        num_bins: int = 5,
        width: int = 1024,
        height: int = 1024,
        prompt: str = "a photo of a person",
        multiplier: float = 1.0,
        seed: Optional[int] = None,
    ) -> ProfileResult:
        """Profile a LoRA file and return per-block per-timestep activity.

        Args:
            lora_path: Path to LoRA safetensors file.
            num_samples: Number of random latents per timestep bin (averaged).
            num_bins: Number of timestep bins (default 5: divides 0.0-1.0 evenly).
            width: Latent width for profiling (1024 recommended for accuracy).
            height: Latent height for profiling.
            prompt: Text prompt — should include the LoRA's trigger word for accurate profiling.
            multiplier: LoRA multiplier.
            seed: Random seed (None for random).

        Returns:
            ProfileResult with activity magnitudes.
        """
        pipeline = self.pipeline
        if not pipeline.is_loaded:
            raise RuntimeError("Pipeline models not loaded.")

        device = pipeline.device

        # Load LoRA in non-merged mode
        pipeline.load_lora_for_profiling(lora_path, multiplier=multiplier)

        # Setup profiling hooks
        pipeline.setup_profiling_hooks()

        # Encode the prompt once
        logger.info(f"Encoding prompt: '{prompt}'")
        ctx_vec, _ = pipeline.encode_prompt(prompt, " ")
        ctx = ctx_vec.to(device=device, dtype=torch.bfloat16)
        ctx, ctx_ids = prc_txt(ctx)

        # Prepare block swap for forward-only
        if hasattr(pipeline.dit, '_offloader') and pipeline.dit._offloader is not None:
            pipeline.dit._offloader.set_forward_only(True)
            pipeline.dit.prepare_block_swap_before_forward()

        # Build timestep bins
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        timestep_bins = [(bin_edges[i], bin_edges[i + 1]) for i in range(num_bins)]

        # Collect block names from first forward pass
        block_names = None

        # Accumulate magnitudes: shape (num_blocks, num_bins)
        all_deltas = None

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        packed_h, packed_w = height // 16, width // 16

        logger.info(f"Profiling: {num_bins} bins x {num_samples} samples at {width}x{height}")

        for bin_idx, (t_low, t_high) in enumerate(tqdm(timestep_bins, desc="Timestep bins")):
            bin_deltas = []

            for sample_idx in range(num_samples):
                # Random timestep within this bin
                t = t_low + (t_high - t_low) * torch.rand(1).item()

                # Generate clean random latent
                clean_latent = torch.randn(
                    1, 128, packed_h, packed_w,
                    generator=generator if seed is not None else None,
                    device=device, dtype=torch.bfloat16,
                )

                # Mix with noise at timestep t (flow matching: x_t = (1-t)*x_0 + t*noise)
                noise = torch.randn_like(clean_latent)
                noisy_latent = (1.0 - t) * clean_latent + t * noise

                # Pack for Klein forward
                x, x_ids = prc_img(noisy_latent)
                t_vec = torch.full((1,), t, device=device, dtype=torch.bfloat16)
                guidance_vec = torch.full((1,), 1.0, device=device, dtype=torch.bfloat16)

                # Forward WITH LoRA
                pipeline.enable_lora()
                pipeline.clear_profiling_buffer()
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    pipeline.dit(
                        x=x, x_ids=x_ids, timesteps=t_vec,
                        ctx=ctx, ctx_ids=ctx_ids, guidance=guidance_vec,
                    )
                with_lora = pipeline.get_profiling_buffer()

                # Forward WITHOUT LoRA (same input)
                pipeline.disable_lora()
                pipeline.clear_profiling_buffer()
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    pipeline.dit(
                        x=x, x_ids=x_ids, timesteps=t_vec,
                        ctx=ctx, ctx_ids=ctx_ids, guidance=guidance_vec,
                    )
                without_lora = pipeline.get_profiling_buffer()

                # Re-enable for next iteration
                pipeline.enable_lora()

                # Get block names from first pass
                if block_names is None:
                    # Sort numerically: double_blocks.0-7, then single_blocks.0-23
                    def _block_sort_key(name):
                        parts = name.split(".")
                        prefix = parts[0]  # "double_blocks" or "single_blocks"
                        idx = int(parts[1])
                        order = 0 if prefix == "double_blocks" else 1
                        return (order, idx)
                    block_names = sorted(with_lora.keys(), key=_block_sort_key)

                # Compute deltas
                deltas = []
                for name in block_names:
                    w = with_lora.get(name, 0.0)
                    wo = without_lora.get(name, 0.0)
                    deltas.append(abs(w - wo))

                bin_deltas.append(deltas)

            # Average across samples for this bin
            avg_deltas = np.mean(bin_deltas, axis=0)

            if all_deltas is None:
                all_deltas = np.zeros((len(block_names), num_bins))
            all_deltas[:, bin_idx] = avg_deltas

        # Cleanup
        pipeline._remove_profiling_hooks()

        result = ProfileResult(
            block_names=block_names,
            timestep_bins=timestep_bins,
            magnitudes=all_deltas,
            metadata={
                "lora_path": lora_path,
                "num_samples": num_samples,
                "num_bins": num_bins,
                "width": width,
                "height": height,
                "prompt": prompt,
                "multiplier": multiplier,
            },
        )

        logger.info(f"Profiling complete: {result.num_blocks} blocks x {result.num_bins} bins")

        # Print quick summary
        top = result.get_top_blocks(5)
        logger.info("Top 5 most active blocks:")
        for name, total in top:
            logger.info(f"  {name}: {total:.4f}")

        return result
