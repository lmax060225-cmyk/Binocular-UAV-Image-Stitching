#!/usr/bin/env python
"""Evaluate cached global registration and optionally re-render warp diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.metadata import read_metadata_csv  # noqa: E402
from src.metrics import projection_rmse, write_projection_rmse_csv  # noqa: E402
from src.optimization_data import load_match_observations  # noqa: E402
from src.pipeline_io import load_pairs, load_transform_array  # noqa: E402
from src.utils import write_json_atomic  # noqa: E402
from src.warping import draw_global_footprints, render_average_preview, transformation_sanity_checks  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--render", action="store_true", help="Re-render projective preview and footprints")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    metadata = read_metadata_csv(config.cache_dir / "image_metadata.csv")
    transforms = load_transform_array(config.cache_dir / "projective_transforms.npy")
    pairs = load_pairs(config.cache_dir / "image_pairs.json")
    observations = load_match_observations(config.cache_dir / "matches", pairs)
    image_paths = {int(row["image_id"]): config.images_dir / row["filename"] for row in metadata}
    image_shapes = {int(row["image_id"]): (int(row["width"]), int(row["height"])) for row in metadata}
    sanity = transformation_sanity_checks(transforms, image_shapes)
    global_rmse, rows = projection_rmse(transforms, observations)
    write_projection_rmse_csv(config.output_dir / "projection_rmse.csv", rows)
    if args.render:
        scale = float((config.raw.get("warp") or {}).get("preview_scale", 0.15))
        render_average_preview(
            transforms, image_paths, image_shapes, config.debug_dir / "projective_preview.jpg", scale=scale
        )
        draw_global_footprints(transforms, image_shapes, config.debug_dir / "projective_footprints.png")

    final_mosaic = config.output_dir / "final_mosaic.jpg"
    final_mask = config.output_dir / "final_mosaic_mask.png"
    mosaic_shape = None
    mask_components = None
    if final_mosaic.is_file():
        mosaic = cv2.imread(str(final_mosaic), cv2.IMREAD_COLOR)
        if mosaic is None:
            raise OSError(f"Could not decode final mosaic: {final_mosaic}")
        mosaic_shape = {"width": int(mosaic.shape[1]), "height": int(mosaic.shape[0])}
    if final_mask.is_file():
        mask = cv2.imread(str(final_mask), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise OSError(f"Could not decode final mask: {final_mask}")
        mask_components = int(cv2.connectedComponents((mask > 0).astype("uint8"))[0] - 1)

    worst = sorted(rows, key=lambda row: float(row["pair_rmse_px"]), reverse=True)[:10]
    summary = {
        "schema_version": 1,
        "image_count": len(transforms),
        "valid_edge_count": len(observations),
        "all_inlier_projection_rmse_px": global_rmse,
        "sanity_warning_count": sum(bool(row["warnings"]) for row in sanity),
        "sanity": sanity,
        "worst_pairs": worst,
        "final_mosaic_shape": mosaic_shape,
        "final_mask_connected_components": mask_components,
    }
    output = config.output_dir / "evaluation_summary.json"
    write_json_atomic(output, summary)
    print("[Evaluation]")
    print(f"Images/valid edges: {len(transforms)}/{len(observations)}")
    print(f"All-inlier projection RMSE: {global_rmse:.3f} px")
    print(f"Transformation warnings: {summary['sanity_warning_count']}")
    print(f"Final mosaic: {mosaic_shape}; mask components: {mask_components}")
    print("Worst pair RMSEs: " + ", ".join(
        f"{row['image_i']}-{row['image_j']}={float(row['pair_rmse_px']):.2f}px" for row in worst
    ))
    print(f"Summary: {output}")
    return 0 if not summary["sanity_warning_count"] and mask_components in {None, 1} else 2


if __name__ == "__main__":
    raise SystemExit(main())
