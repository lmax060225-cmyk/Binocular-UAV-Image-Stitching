"""Registration metrics computed from all RANSAC inlier matches."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .optimization_data import MatchObservation
from .transforms import warp_points
from .utils import write_csv_atomic


def projection_rmse(
    transforms: dict[int, np.ndarray],
    observations: list[MatchObservation],
) -> tuple[float, list[dict[str, float | int]]]:
    """Compute global projection RMSE using every SIFT+RANSAC inlier.

    The selected P optimization points are deliberately not used here. Pair
    RMSE is the square root of mean squared Euclidean global-pixel distance;
    global RMSE pools squared distances over all pairs and inliers.
    """

    rows: list[dict[str, float | int]] = []
    squared_chunks = []
    for item in observations:
        first = warp_points(transforms[item.image_i], item.all_pts_i)
        second = warp_points(transforms[item.image_j], item.all_pts_j)
        squared = np.sum((first - second) ** 2, axis=1)
        squared_chunks.append(squared)
        rows.append(
            {
                "image_i": item.image_i,
                "image_j": item.image_j,
                "inlier_count": int(squared.size),
                "pair_rmse_px": float(np.sqrt(np.mean(squared))),
                "pair_median_error_px": float(np.sqrt(np.median(squared))),
                "pair_max_error_px": float(np.sqrt(np.max(squared))),
            }
        )
    if not squared_chunks:
        raise ValueError("No observations supplied for projection RMSE")
    all_squared = np.concatenate(squared_chunks)
    return float(np.sqrt(np.mean(all_squared))), rows


def write_projection_rmse_csv(path: str | Path, rows: list[dict[str, float | int]]) -> None:
    """Write per-pair projection metrics atomically."""

    write_csv_atomic(
        Path(path), rows,
        ("image_i", "image_j", "inlier_count", "pair_rmse_px", "pair_median_error_px", "pair_max_error_px"),
    )
