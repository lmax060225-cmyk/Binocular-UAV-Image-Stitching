"""GPS-pair-restricted SIFT matching, RANSAC filtering, and cache I/O."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .features import FeatureSet
from .transforms import warp_points
from .uniform_sampling import select_top_p_matches, select_uniform_matches
from .utils import ensure_parent


@dataclass(frozen=True)
class PairMatch:
    """All ratio-test matches, RANSAC mask, and optimization subset for one pair."""

    image_i: int
    image_j: int
    pts_i: np.ndarray
    pts_j: np.ndarray
    distances: np.ndarray
    inlier_mask: np.ndarray
    homography_i_to_j: np.ndarray
    selected_inlier_indices: np.ndarray
    raw_match_count: int
    ratio_match_count: int
    ratio_test: float
    ransac_threshold_px: float
    min_inliers: int
    selection_method: str
    selected_points: int
    grid_rows: int
    grid_cols: int

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def valid(self) -> bool:
        return bool(self.inlier_count >= self.min_inliers and np.all(np.isfinite(self.homography_i_to_j)))

    @property
    def inlier_pts_i(self) -> np.ndarray:
        return self.pts_i[self.inlier_mask]

    @property
    def inlier_pts_j(self) -> np.ndarray:
        return self.pts_j[self.inlier_mask]

    @property
    def inlier_distances(self) -> np.ndarray:
        return self.distances[self.inlier_mask]

    @property
    def selected_pts_i(self) -> np.ndarray:
        return self.inlier_pts_i[self.selected_inlier_indices]

    @property
    def selected_pts_j(self) -> np.ndarray:
        return self.inlier_pts_j[self.selected_inlier_indices]

    @property
    def inlier_rmse_px(self) -> float:
        if not self.valid:
            return float("inf")
        residual = warp_points(self.homography_i_to_j, self.inlier_pts_i) - self.inlier_pts_j
        return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def match_feature_pair(
    features_i: FeatureSet,
    features_j: FeatureSet,
    *,
    ratio_test: float,
    ransac_threshold_px: float,
    min_inliers: int,
    selection_method: str,
    selected_points: int,
    grid_rows: int,
    grid_cols: int,
) -> PairMatch:
    """Match one GPS-neighbor pair and select paper-style optimization points."""

    if not 0.0 < ratio_test < 1.0:
        raise ValueError("ratio_test must be between 0 and 1")
    if features_i.descriptors.shape[0] < 2 or features_j.descriptors.shape[0] < 2:
        return _empty_match(
            features_i.image_id, features_j.image_id, ratio_test, ransac_threshold_px,
            min_inliers, selection_method, selected_points, grid_rows, grid_cols,
        )
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    neighbors = matcher.knnMatch(features_i.descriptors, features_j.descriptors, k=2)
    accepted = [first for pair in neighbors if len(pair) == 2 for first, second in [pair] if first.distance < ratio_test * second.distance]
    if accepted:
        pts_i = np.asarray([features_i.xy[item.queryIdx] for item in accepted], dtype=np.float32)
        pts_j = np.asarray([features_j.xy[item.trainIdx] for item in accepted], dtype=np.float32)
        distances = np.asarray([item.distance for item in accepted], dtype=np.float32)
    else:
        pts_i = pts_j = np.empty((0, 2), dtype=np.float32)
        distances = np.empty((0,), dtype=np.float32)

    homography = np.full((3, 3), np.nan, dtype=np.float64)
    inlier_mask = np.zeros((pts_i.shape[0],), dtype=bool)
    if pts_i.shape[0] >= 4:
        estimated, mask = cv2.findHomography(pts_i, pts_j, cv2.RANSAC, float(ransac_threshold_px))
        if estimated is not None and mask is not None:
            homography = np.asarray(estimated, dtype=np.float64)
            inlier_mask = mask.reshape(-1).astype(bool)

    inlier_distances = distances[inlier_mask]
    if np.count_nonzero(inlier_mask) >= min_inliers:
        if selection_method.casefold() == "uniform":
            selected = select_uniform_matches(
                pts_i[inlier_mask], pts_j[inlier_mask], inlier_distances,
                selected_points=selected_points, grid_rows=grid_rows, grid_cols=grid_cols,
            )
        elif selection_method.casefold() in {"top_p", "top-p", "top"}:
            selected = select_top_p_matches(inlier_distances, selected_points)
        else:
            raise ValueError(f"Unknown sampling method: {selection_method}")
    else:
        selected = np.empty((0,), dtype=np.int32)
    return PairMatch(
        image_i=features_i.image_id,
        image_j=features_j.image_id,
        pts_i=pts_i,
        pts_j=pts_j,
        distances=distances,
        inlier_mask=inlier_mask,
        homography_i_to_j=homography,
        selected_inlier_indices=selected,
        raw_match_count=len(neighbors),
        ratio_match_count=len(accepted),
        ratio_test=float(ratio_test),
        ransac_threshold_px=float(ransac_threshold_px),
        min_inliers=int(min_inliers),
        selection_method=selection_method.casefold(),
        selected_points=int(selected_points),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
    )


def save_pair_match(path: str | Path, match: PairMatch) -> None:
    """Atomically persist one pair cache."""

    destination = Path(path)
    ensure_parent(destination)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                schema_version=np.int32(1), image_i=np.int32(match.image_i), image_j=np.int32(match.image_j),
                pts_i=match.pts_i, pts_j=match.pts_j, distances=match.distances,
                inlier_mask=match.inlier_mask.astype(np.uint8), homography_i_to_j=match.homography_i_to_j,
                selected_inlier_indices=match.selected_inlier_indices,
                raw_match_count=np.int32(match.raw_match_count), ratio_match_count=np.int32(match.ratio_match_count),
                ratio_test=np.float64(match.ratio_test), ransac_threshold_px=np.float64(match.ransac_threshold_px),
                min_inliers=np.int32(match.min_inliers), selection_method=np.asarray(match.selection_method),
                selected_points=np.int32(match.selected_points), grid_rows=np.int32(match.grid_rows), grid_cols=np.int32(match.grid_cols),
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_pair_match(path: str | Path) -> PairMatch:
    """Load one match cache without pickle."""

    with np.load(Path(path), allow_pickle=False) as payload:
        return PairMatch(
            image_i=int(payload["image_i"]), image_j=int(payload["image_j"]),
            pts_i=payload["pts_i"].astype(np.float32, copy=False), pts_j=payload["pts_j"].astype(np.float32, copy=False),
            distances=payload["distances"].astype(np.float32, copy=False),
            inlier_mask=payload["inlier_mask"].astype(bool),
            homography_i_to_j=payload["homography_i_to_j"].astype(np.float64, copy=False),
            selected_inlier_indices=payload["selected_inlier_indices"].astype(np.int32, copy=False),
            raw_match_count=int(payload["raw_match_count"]), ratio_match_count=int(payload["ratio_match_count"]),
            ratio_test=float(payload["ratio_test"]), ransac_threshold_px=float(payload["ransac_threshold_px"]),
            min_inliers=int(payload["min_inliers"]), selection_method=str(payload["selection_method"]),
            selected_points=int(payload["selected_points"]), grid_rows=int(payload["grid_rows"]), grid_cols=int(payload["grid_cols"]),
        )


def cache_matches_configuration(match: PairMatch, *, ratio_test: float, ransac_threshold_px: float, min_inliers: int,
                                selection_method: str, selected_points: int, grid_rows: int, grid_cols: int) -> bool:
    """Return whether a cached match was produced by the current parameters."""

    return (
        abs(match.ratio_test - ratio_test) < 1e-12
        and abs(match.ransac_threshold_px - ransac_threshold_px) < 1e-12
        and match.min_inliers == min_inliers
        and match.selection_method == selection_method.casefold()
        and match.selected_points == selected_points
        and match.grid_rows == grid_rows
        and match.grid_cols == grid_cols
    )


def _empty_match(image_i: int, image_j: int, ratio_test: float, ransac_threshold_px: float, min_inliers: int,
                 selection_method: str, selected_points: int, grid_rows: int, grid_cols: int) -> PairMatch:
    return PairMatch(
        image_i=image_i, image_j=image_j, pts_i=np.empty((0, 2), np.float32), pts_j=np.empty((0, 2), np.float32),
        distances=np.empty((0,), np.float32), inlier_mask=np.empty((0,), bool),
        homography_i_to_j=np.full((3, 3), np.nan), selected_inlier_indices=np.empty((0,), np.int32),
        raw_match_count=0, ratio_match_count=0, ratio_test=ratio_test, ransac_threshold_px=ransac_threshold_px,
        min_inliers=min_inliers, selection_method=selection_method.casefold(), selected_points=selected_points,
        grid_rows=grid_rows, grid_cols=grid_cols,
    )
