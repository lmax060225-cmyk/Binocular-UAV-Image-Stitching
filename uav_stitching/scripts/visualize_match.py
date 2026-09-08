#!/usr/bin/env python
"""Regenerate the RANSAC/uniform-selection visualization for one cached pair."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.matching import load_pair_match  # noqa: E402
from src.metadata import read_metadata_csv  # noqa: E402
from src.visualization import draw_uniform_match_debug  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pair", required=True, nargs=2, type=int, metavar=("I", "J"))
    args = parser.parse_args()
    config = load_config(args.config)
    image_i, image_j = sorted(args.pair)
    match = load_pair_match(config.cache_dir / "matches" / f"{image_i:04d}_{image_j:04d}.npz")
    metadata = {int(row["image_id"]): row for row in read_metadata_csv(config.cache_dir / "image_metadata.csv")}
    output = config.debug_dir / f"uniform_matches_{image_i}_{image_j}.jpg"
    draw_uniform_match_debug(
        config.images_dir / metadata[image_i]["filename"], config.images_dir / metadata[image_j]["filename"],
        match.inlier_pts_i, match.inlier_pts_j, match.selected_inlier_indices, output,
        grid_rows=match.grid_rows, grid_cols=match.grid_cols,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
