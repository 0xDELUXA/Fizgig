"""CLI entry point for Klein 9B LoRA training.

Usage:
    accelerate launch src/fizgig/scripts/train.py --dit path/to/dit --vae path/to/ae ...

This is the script the GUI calls via subprocess.
"""

import sys
import os

# Ensure fizgig package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fizgig.training.trainer import KleinTrainer, setup_parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    trainer = KleinTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
