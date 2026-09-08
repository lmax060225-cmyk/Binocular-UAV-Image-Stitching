#!/usr/bin/env python
"""Phase 4: cache SIFT features and match only GPS-neighbor image pairs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.features import FeatureSet, load_or_extract_features  # noqa: E402
from src.matching import (  # noqa: E402
    cache_matches_configuration,
    load_pair_match,
    match_feature_pair,
    save_pair_match,
)
from src.metadata import read_metadata_csv  # noqa: E402
from src.utils import configure_logging, write_json_atomic  # noqa: E402
from src.visualization import draw_uniform_match_debug  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pair", nargs=2, type=int, action="append", metavar=("I", "J"),
                        help="Process only a specified internal image-ID pair; repeatable")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-matches", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    feature_config = config.raw.get("features") or {}
    sampling_config = config.raw.get("sampling") or {}
    nfeatures = int(feature_config.get("nfeatures", 8000))
    ratio_test = float(feature_config.get("ratio_test", 0.75))
    ransac_threshold = float(feature_config.get("ransac_threshold_px", 4.0))
    min_inliers = int(feature_config.get("min_inliers", 20))
    selection_method = str(sampling_config.get("method", "uniform"))
    selected_points = int(sampling_config.get("selected_points", 40))
    grid_rows = int(sampling_config.get("grid_rows", 5))
    grid_cols = int(sampling_config.get("grid_cols", 8))

    metadata = read_metadata_csv(config.cache_dir / "image_metadata.csv")
    image_paths = {int(row["image_id"]): config.images_dir / row["filename"] for row in metadata}
    with (config.cache_dir / "image_pairs.json").open("r", encoding="utf-8") as stream:
        all_pairs = [tuple(map(int, pair)) for pair in json.load(stream)["pairs"]]
    if args.pair:
        requested = {tuple(sorted(pair)) for pair in args.pair}
        missing = requested - set(all_pairs)
        if missing:
            raise ValueError(f"Requested pair(s) are not in the GPS graph: {sorted(missing)}")
        pairs = sorted(requested)
    else:
        pairs = all_pairs
    needed_ids = sorted({image_id for pair in pairs for image_id in pair})

    feature_dir = config.cache_dir / "features"
    features: dict[int, FeatureSet] = {}
    extracted_ids: set[int] = set()
    print("[Features]")
    feature_start = time.perf_counter()
    for image_id in tqdm(needed_ids, desc="SIFT", unit="image"):
        feature_set, extracted = load_or_extract_features(
            image_id, image_paths[image_id], feature_dir / f"{image_id:04d}.npz",
            nfeatures=nfeatures, force=args.force_features,
        )
        features[image_id] = feature_set
        if extracted:
            extracted_ids.add(image_id)
        tqdm.write(f"Image {image_id}: {feature_set.count} keypoints ({'extracted' if extracted else 'cache'})")
    print(f"Feature time: {time.perf_counter() - feature_start:.2f}s; extracted={len(extracted_ids)} cached={len(needed_ids)-len(extracted_ids)}")

    match_dir = config.cache_dir / "matches"
    valid_count = 0
    recomputed = 0
    summary_rows = []
    print("\n[Matching]")
    match_start = time.perf_counter()
    for image_i, image_j in tqdm(pairs, desc="Pairs", unit="pair"):
        cache_path = match_dir / f"{image_i:04d}_{image_j:04d}.npz"
        match = None
        if cache_path.is_file() and not args.force_matches and image_i not in extracted_ids and image_j not in extracted_ids:
            try:
                candidate = load_pair_match(cache_path)
                if cache_matches_configuration(
                    candidate, ratio_test=ratio_test, ransac_threshold_px=ransac_threshold,
                    min_inliers=min_inliers, selection_method=selection_method,
                    selected_points=selected_points, grid_rows=grid_rows, grid_cols=grid_cols,
                ):
                    match = candidate
            except (OSError, ValueError, KeyError):
                match = None
        if match is None:
            match = match_feature_pair(
                features[image_i], features[image_j], ratio_test=ratio_test,
                ransac_threshold_px=ransac_threshold, min_inliers=min_inliers,
                selection_method=selection_method, selected_points=selected_points,
                grid_rows=grid_rows, grid_cols=grid_cols,
            )
            save_pair_match(cache_path, match)
            recomputed += 1
        valid_count += int(match.valid)
        rmse = match.inlier_rmse_px if match.valid else None
        summary_rows.append({
            "image_i": image_i, "image_j": image_j, "raw": match.raw_match_count,
            "ratio": match.ratio_match_count, "inliers": match.inlier_count,
            "selected": int(match.selected_inlier_indices.size), "valid": match.valid,
            "pairwise_homography_rmse_px": rmse,
        })
        tqdm.write(
            f"{image_i} <-> {image_j}: raw={match.raw_match_count} ratio={match.ratio_match_count} "
            f"ransac={match.inlier_count} selected={match.selected_inlier_indices.size} valid={match.valid}"
        )
        if args.pair and match.valid:
            debug_path = config.debug_dir / f"uniform_matches_{image_i}_{image_j}.jpg"
            draw_uniform_match_debug(
                image_paths[image_i], image_paths[image_j], match.inlier_pts_i, match.inlier_pts_j,
                match.selected_inlier_indices, debug_path, grid_rows=grid_rows, grid_cols=grid_cols,
            )

    summary = {
        "schema_version": 1, "pair_count": len(pairs), "valid_pairs": valid_count,
        "invalid_pairs": len(pairs) - valid_count, "recomputed_pairs": recomputed,
        "elapsed_seconds": time.perf_counter() - match_start, "pairs": summary_rows,
    }
    write_json_atomic(config.cache_dir / "matching_summary.json", summary)
    print(f"Valid pairs: {valid_count}/{len(pairs)}")
    print(f"Match time: {summary['elapsed_seconds']:.2f}s; recomputed={recomputed}")
    print(f"Summary: {config.cache_dir / 'matching_summary.json'}")
    return 0 if valid_count else 2


if __name__ == "__main__":
    raise SystemExit(main())
