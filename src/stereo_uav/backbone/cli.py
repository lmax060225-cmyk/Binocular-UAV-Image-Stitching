"""Command-line interface for block-incremental stereo registration."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from . import config
from .pipeline import run_stitching


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, default=config.LEFT_IN)
    parser.add_argument("--right", type=Path, default=config.RIGHT_IN)
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument(
        "--graphcut",
        action="store_true",
        help="Use Graph-Cut seams instead of ordered pixel overwrite",
    )
    parser.add_argument("--opencv-threads", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if not args.left.is_dir() or not args.right.is_dir():
        raise SystemExit("Both --left and --right must be existing image directories.")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Output directory is not empty; select a new --output directory.")
    previous = (config.LEFT_IN, config.RIGHT_IN, config.OUTPUT_DIR, config.USE_GRAPHCUT)
    try:
        config.LEFT_IN, config.RIGHT_IN, config.OUTPUT_DIR = args.left, args.right, args.output
        config.USE_GRAPHCUT = args.graphcut
        cv2.setNumThreads(args.opencv_threads)
        cv2.setRNGSeed(config.RANDOM_SEED)
        run_stitching()
    finally:
        config.LEFT_IN, config.RIGHT_IN, config.OUTPUT_DIR, config.USE_GRAPHCUT = previous
    return 0
