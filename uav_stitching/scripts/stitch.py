#!/usr/bin/env python
"""Phase 9: sequentially Graph-Cut blend globally registered UAV images."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.gps import choose_center_reference, read_positions_csv  # noqa: E402
from src.graphcut_blending import sequential_graphcut_blend, strongest_overlap_order  # noqa: E402
from src.metadata import read_metadata_csv  # noqa: E402
from src.optimization_data import filter_observations, load_match_observations  # noqa: E402
from src.pipeline_io import load_pairs, load_transform_array, select_connected_subset  # noqa: E402
from src.utils import write_json_atomic  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--subset-size", type=int, help="Blend a connected real-image subset")
    parser.add_argument("--output-scale", type=float, help="Override final mosaic scale")
    parser.add_argument("--seam-scale", type=float, help="Override Graph-Cut working scale")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    warp_config = config.raw.get("warp") or {}
    blend_config = config.raw.get("blend") or {}
    if str(blend_config.get("method", "graphcut")).casefold() != "graphcut":
        raise ValueError("Only blend.method=graphcut is implemented")
    if bool(blend_config.get("exposure_compensation", False)):
        raise ValueError("Exposure compensation is optional but not enabled in this paper-faithful run")

    metadata = read_metadata_csv(config.cache_dir / "image_metadata.csv")
    positions = read_positions_csv(config.cache_dir / "image_positions.csv")
    all_ids = sorted(int(row["image_id"]) for row in metadata)
    reference = choose_center_reference(positions)
    pairs = load_pairs(config.cache_dir / "image_pairs.json")
    all_observations = load_match_observations(config.cache_dir / "matches", pairs)
    image_ids = select_connected_subset(all_ids, all_observations, reference, args.subset_size)
    observations = filter_observations(all_observations, set(image_ids))
    suffix = f"_small{len(image_ids)}" if len(image_ids) < len(all_ids) else ""
    transforms = load_transform_array(config.cache_dir / f"projective_transforms{suffix}.npy")
    transforms = {image_id: transforms[image_id] for image_id in image_ids}
    order = strongest_overlap_order(image_ids, observations, reference)
    image_paths = {int(row["image_id"]): config.images_dir / row["filename"] for row in metadata}
    image_shapes = {int(row["image_id"]): (int(row["width"]), int(row["height"])) for row in metadata}
    output_scale = args.output_scale if args.output_scale is not None else float(warp_config.get("final_scale", 0.25))
    seam_scale = args.seam_scale if args.seam_scale is not None else float(blend_config.get("seam_scale", 0.10))
    max_pixels = int(warp_config.get("max_final_pixels", 50_000_000))

    print("[Warp/Blend]")
    print(f"Images: {len(image_ids)}; reference: {reference}")
    print(f"Order: {order}")
    start = time.perf_counter()
    result = sequential_graphcut_blend(
        transforms,
        image_paths,
        image_shapes,
        order,
        config.output_dir / f"final_mosaic{suffix}.jpg",
        config.output_dir / f"final_mosaic_mask{suffix}.png",
        config.debug_dir / f"seams{suffix}",
        output_scale=output_scale,
        seam_scale=seam_scale,
        max_output_pixels=max_pixels,
    )
    elapsed = time.perf_counter() - start
    summary = {
        "schema_version": 1,
        "method": "opencv_graphcut_engineering_equivalent",
        "frame_to_frame": True,
        "exposure_compensation": False,
        "image_count": len(image_ids),
        "reference_image": reference,
        "order": list(result.order),
        "full_resolution_canvas": {
            "width": result.full_resolution_geometry.width,
            "height": result.full_resolution_geometry.height,
        },
        "output_canvas": {"width": result.output_width, "height": result.output_height},
        "output_scale": result.output_scale,
        "seam_scale": result.seam_scale,
        "covered_pixels": result.covered_pixels,
        "coverage_ratio": result.coverage_ratio,
        "elapsed_seconds": elapsed,
        "steps": [
            {
                "step": item.step,
                "image_id": item.image_id,
                "overlap_pixels_at_seam_scale": item.overlap_pixels_at_seam_scale,
                "seam_method": item.seam_method,
                "elapsed_seconds": item.elapsed_seconds,
            }
            for item in result.steps
        ],
    }
    summary_path = config.output_dir / f"stitch_summary{suffix}.json"
    write_json_atomic(summary_path, summary)
    fallback_steps = [item.step for item in result.steps if item.seam_method not in {"initial", "opencv_graphcut"}]
    print(
        f"Full/output canvas: {result.full_resolution_geometry.width}x{result.full_resolution_geometry.height} "
        f"-> {result.output_width}x{result.output_height}"
    )
    print(f"Estimated output RGB+mask memory: {(result.output_width * result.output_height * 4) / 2**20:.1f} MiB")
    print(f"Coverage: {result.coverage_ratio:.3%}; fallback steps: {fallback_steps}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Final mosaic: {result.output_path}")
    print(f"Final mask: {result.mask_path}")
    print(f"Summary: {summary_path}")
    return 0 if not fallback_steps else 2


if __name__ == "__main__":
    raise SystemExit(main())
