#!/usr/bin/env python
"""Phase 5: solve the shared global affine registration problem."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.affine_optimizer import optimize_affines  # noqa: E402
from src.config import load_config  # noqa: E402
from src.gps import choose_center_reference, read_positions_csv  # noqa: E402
from src.metadata import read_metadata_csv  # noqa: E402
from src.metrics import projection_rmse  # noqa: E402
from src.optimization_data import (  # noqa: E402
    assert_observation_graph_connected,
    filter_observations,
    load_match_observations,
)
from src.pipeline_io import load_pairs, save_transform_array, select_connected_subset  # noqa: E402
from src.utils import write_json_atomic  # noqa: E402
from src.warping import draw_global_footprints, render_average_preview, transformation_sanity_checks  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--subset-size", type=int, help="Connected real-image smoke test size")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), help="Override SciPy verbosity")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    optimization = config.raw.get("optimization") or {}
    preview_scale = float((config.raw.get("warp") or {}).get("preview_scale", 0.15))
    metadata = read_metadata_csv(config.cache_dir / "image_metadata.csv")
    positions = read_positions_csv(config.cache_dir / "image_positions.csv")
    all_ids = sorted(int(row["image_id"]) for row in metadata)
    reference = choose_center_reference(positions)
    pairs = load_pairs(config.cache_dir / "image_pairs.json")
    all_observations = load_match_observations(config.cache_dir / "matches", pairs)
    image_ids = select_connected_subset(all_ids, all_observations, reference, args.subset_size)
    observations = filter_observations(all_observations, set(image_ids))
    assert_observation_graph_connected(image_ids, observations)

    print(f"Reference image: {reference}")
    print(f"Images/valid edges: {len(image_ids)}/{len(observations)}")
    start = time.perf_counter()
    result = optimize_affines(
        image_ids,
        observations,
        reference,
        sigma_tr=float(optimization.get("sigma_tr", 5000.0)),
        max_nfev=int(optimization.get("max_nfev_affine", 200)),
        verbose=args.verbose if args.verbose is not None else int(optimization.get("verbose", 0)),
    )
    elapsed = time.perf_counter() - start
    suffix = f"_small{len(image_ids)}" if len(image_ids) < len(all_ids) else ""
    transform_path = config.cache_dir / f"affine_transforms{suffix}.npy"
    save_transform_array(transform_path, result.transforms, len(all_ids))

    image_paths = {int(row["image_id"]): config.images_dir / row["filename"] for row in metadata}
    image_shapes = {int(row["image_id"]): (int(row["width"]), int(row["height"])) for row in metadata}
    geometry = render_average_preview(
        result.transforms, image_paths, image_shapes, config.debug_dir / f"affine_preview{suffix}.jpg",
        scale=preview_scale,
    )
    draw_global_footprints(result.transforms, image_shapes, config.debug_dir / f"affine_footprints{suffix}.png")
    sanity = transformation_sanity_checks(result.transforms, image_shapes)
    global_rmse, _ = projection_rmse(result.transforms, observations)
    summary = {
        "schema_version": 1,
        "stage": "affine",
        "image_count": len(image_ids),
        "valid_edge_count": len(observations),
        "reference_image": reference,
        "initial_cost": result.initial_cost,
        "final_cost": result.final_cost,
        "initial_selected_rmse_px": result.initial_rmse_px,
        "final_selected_rmse_px": result.final_rmse_px,
        "all_inlier_projection_rmse_px": global_rmse,
        "success": result.success,
        "status": result.status,
        "message": result.message,
        "nfev": result.nfev,
        "optimality": result.optimality,
        "elapsed_seconds": elapsed,
        "full_resolution_canvas": {"width": geometry.width, "height": geometry.height},
        "sanity_warning_count": sum(bool(row["warnings"]) for row in sanity),
        "sanity": sanity,
    }
    summary_path = config.cache_dir / f"affine_summary{suffix}.json"
    write_json_atomic(summary_path, summary)

    print(f"Selected-point RMSE: {result.initial_rmse_px:.3f} -> {result.final_rmse_px:.3f} px")
    print(f"All-inlier projection RMSE: {global_rmse:.3f} px")
    print(f"Canvas: {geometry.width} x {geometry.height}; sanity warnings={summary['sanity_warning_count']}")
    print(f"Elapsed: {elapsed:.2f}s; success={result.success}; nfev={result.nfev}")
    print(f"Transforms: {transform_path}")
    print(f"Summary: {summary_path}")
    for image_id, matrix in sorted(result.transforms.items()):
        print(f"H_affine[{image_id}] =\n{matrix}")
    return 0 if result.success and not summary["sanity_warning_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
