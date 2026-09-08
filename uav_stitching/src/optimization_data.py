"""Validated match observations consumed by both global optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .matching import load_pair_match


@dataclass(frozen=True)
class MatchObservation:
    """Selected optimization points plus all RANSAC inliers for one image pair."""

    image_i: int
    image_j: int
    selected_pts_i: np.ndarray
    selected_pts_j: np.ndarray
    all_pts_i: np.ndarray
    all_pts_j: np.ndarray
    inlier_count: int
    pairwise_homography_i_to_j: np.ndarray
    pairwise_rmse_px: float


def load_match_observations(match_dir: str | Path, pairs: list[tuple[int, int]]) -> list[MatchObservation]:
    """Load every valid cached pair and omit explicitly invalid matches."""

    directory = Path(match_dir)
    observations: list[MatchObservation] = []
    for image_i, image_j in pairs:
        match = load_pair_match(directory / f"{image_i:04d}_{image_j:04d}.npz")
        if not match.valid or match.selected_inlier_indices.size == 0:
            continue
        observations.append(
            MatchObservation(
                image_i=image_i,
                image_j=image_j,
                selected_pts_i=match.selected_pts_i.astype(np.float64),
                selected_pts_j=match.selected_pts_j.astype(np.float64),
                all_pts_i=match.inlier_pts_i.astype(np.float64),
                all_pts_j=match.inlier_pts_j.astype(np.float64),
                inlier_count=match.inlier_count,
                pairwise_homography_i_to_j=match.homography_i_to_j.astype(np.float64),
                pairwise_rmse_px=match.inlier_rmse_px,
            )
        )
    return observations


def observation_image_ids(observations: list[MatchObservation]) -> list[int]:
    """Return sorted unique image IDs represented by observations."""

    return sorted({image_id for item in observations for image_id in (item.image_i, item.image_j)})


def filter_observations(observations: list[MatchObservation], image_ids: set[int]) -> list[MatchObservation]:
    """Keep only observations whose two endpoints are in a requested subset."""

    return [item for item in observations if item.image_i in image_ids and item.image_j in image_ids]


def assert_observation_graph_connected(image_ids: list[int], observations: list[MatchObservation]) -> None:
    """Raise if selected match observations do not connect all requested images."""

    if not image_ids:
        raise ValueError("No image IDs supplied")
    adjacency = {image_id: set() for image_id in image_ids}
    for item in observations:
        if item.image_i in adjacency and item.image_j in adjacency:
            adjacency[item.image_i].add(item.image_j)
            adjacency[item.image_j].add(item.image_i)
    visited = {image_ids[0]}
    stack = [image_ids[0]]
    while stack:
        current = stack.pop()
        unseen = adjacency[current] - visited
        visited.update(unseen)
        stack.extend(unseen)
    missing = sorted(set(image_ids) - visited)
    if missing:
        raise ValueError(f"Valid match graph is disconnected; unreachable image IDs: {missing}")
