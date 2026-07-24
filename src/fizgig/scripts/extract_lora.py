"""CLI entry point for LoRA extraction.

Usage:
    python src/fizgig/scripts/extract_lora.py \
      --source path/to/source.safetensors \
      --output path/to/extracted.safetensors \
      --dit path/to/klein-9b \
      --vae path/to/vae \
      --text_encoder path/to/qwen3 \
      --blocks style \
      --rank 2
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fizgig.klein.inference import KleinInferencePipeline
from fizgig.extraction.extractor import LoRAExtractor, ExtractionConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


TIMESTEP_PRESETS = {
    "all": (0.0, 1.0),
    "early": (0.6, 1.0),
    "mid": (0.3, 0.7),
    "late": (0.0, 0.4),
    "midlate": (0.0, 0.7),
    "earlymid": (0.3, 1.0),
}


def main():
    parser = argparse.ArgumentParser(description="Extract a low-rank LoRA from an existing LoRA")
    parser.add_argument("--source", type=str, required=True, help="Source LoRA safetensors path")
    parser.add_argument("--output", type=str, required=True, help="Output LoRA safetensors path")
    parser.add_argument("--dit", type=str, default=None, help="Klein DiT checkpoint (required unless --samples 0)")
    parser.add_argument("--vae", type=str, default=None, help="VAE/AE checkpoint (required unless --samples 0)")
    parser.add_argument("--text_encoder", type=str, default=None, help="Qwen3-8B checkpoint (required unless --samples 0)")
    parser.add_argument("--rank", type=int, default=2, help="Target rank (default: 2)")
    parser.add_argument("--blocks", type=str, default="all",
                        help="Comma-separated block categories: all, style_composition, identity, details, or 'custom' with --custom_blocks")
    parser.add_argument("--custom_blocks", type=str, default=None,
                        help="Comma-separated block names (e.g. 'double_blocks.5,double_blocks.6,single_blocks.12'). Used when --blocks=custom.")
    parser.add_argument("--timesteps", type=str, default="all",
                        choices=list(TIMESTEP_PRESETS.keys()),
                        help="Timestep range preset")
    parser.add_argument("--samples", type=int, default=16, help="Number of samples (default: 16)")
    parser.add_argument("--prompt", type=str, default="a photo", help="Conditioning prompt")
    parser.add_argument("--width", type=int, default=1024, help="Width (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Height (default: 1024)")
    parser.add_argument("--multiplier", type=float, default=1.0, help="Source LoRA multiplier")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--fp8_text_encoder", action="store_true", help="Use FP8 text encoder")
    args = parser.parse_args()

    blocks = [b.strip() for b in args.blocks.split(",")]
    custom_blocks = None
    if "custom" in blocks:
        if not args.custom_blocks:
            parser.error("--blocks=custom requires --custom_blocks to be set")
        custom_blocks = [b.strip() for b in args.custom_blocks.split(",")]
    timestep_range = TIMESTEP_PRESETS[args.timesteps]

    config = ExtractionConfig(
        source_lora_path=args.source,
        output_lora_path=args.output,
        target_rank=args.rank,
        timestep_range=timestep_range,
        include_blocks=blocks,
        custom_blocks=custom_blocks,
        num_samples=args.samples,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        source_multiplier=args.multiplier,
        seed=args.seed,
    )

    if args.samples == 0:
        # Pure weight SVD: no pipeline, no GPU models — model-agnostic, so this
        # path handles Krea 2 LoRAs as well as Klein (use --blocks all for
        # Krea 2; the block categories are Klein's semantic map).
        result = LoRAExtractor.extract_weight_only(config)
    else:
        if not (args.dit and args.vae and args.text_encoder):
            parser.error("--dit, --vae and --text_encoder are required for activation-weighted "
                         "extraction (--samples > 0); pass --samples 0 for weight-only SVD.")

        # Auto-detect model version and fp8 from filename
        dit_basename = os.path.basename(args.dit).lower()
        model_version = "klein-base-9b" if "base" in dit_basename else "klein-9b"
        is_fp8_model = "fp8" in dit_basename

        pipeline = KleinInferencePipeline()
        pipeline.load_models(
            dit_path=args.dit,
            vae_path=args.vae,
            text_encoder_path=args.text_encoder,
            model_version=model_version,
            device="cuda",
            fp8_scaled=not is_fp8_model,
            fp8_text_encoder=args.fp8_text_encoder,
            blocks_to_swap=0,
        )

        extractor = LoRAExtractor(pipeline)
        result = extractor.extract(config)

    print(f"\nExtraction complete:")
    print(f"  Output: {result.output_path}")
    print(f"  Layers: {result.num_layers_extracted}")
    print(f"  Rank: {result.target_rank}")
    print(f"  Params: {result.total_params:,}")
    print(f"  Time: {result.elapsed_seconds:.1f}s")

    if args.samples > 0:
        pipeline.unload_models()


if __name__ == "__main__":
    main()
