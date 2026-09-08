"""Evenly distributed match selection from RANSAC inliers."""

from __future__ import annotations

import numpy as np


def select_uniform_matches(
    pts_i: np.ndarray,
    pts_j: np.ndarray,
    distances: np.ndarray,
    *,
    selected_points: int,
    grid_rows: int,
    grid_cols: int,
) -> np.ndarray:
    """Return indices that cover the overlap bounding box in image ``i``.

    One minimum-descriptor-distance match is selected from every non-empty
    cell. If fewer than ``selected_points`` cells are occupied, remaining
    points are added by deterministic farthest-point sampling in normalized
    overlap coordinates. The returned indices address the supplied RANSAC
    inlier arrays and never exceed ``selected_points``.
    """

    points_i = np.asarray(pts_i, dtype=np.float64)
    points_j = np.asarray(pts_j, dtype=np.float64)
    scores = np.asarray(distances, dtype=np.float64).reshape(-1)
    if points_i.shape != points_j.shape or points_i.ndim != 2 or points_i.shape[1] != 2:
        raise ValueError("pts_i and pts_j must have matching shape (N,2)")
    if scores.shape[0] != points_i.shape[0]:
        raise ValueError("distances length must equal point count")
    if selected_points <= 0 or grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("selected_points and grid dimensions must be positive")
    count = points_i.shape[0]
    if count == 0:
        return np.empty((0,), dtype=np.int32)

    minimum = points_i.min(axis=0)
    maximum = points_i.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-9)
    normalized = np.clip((points_i - minimum) / span, 0.0, 1.0)
    columns = np.minimum((normalized[:, 0] * grid_cols).astype(np.int32), grid_cols - 1)
    rows = np.minimum((normalized[:, 1] * grid_rows).astype(np.int32), grid_rows - 1)

    selected: list[int] = []
    for row in range(grid_rows):
        for column in range(grid_cols):
            candidates = np.flatnonzero((rows == row) & (columns == column))
            if candidates.size:
                best = min(candidates.tolist(), key=lambda index: (scores[index], index))
                selected.append(best)
    if len(selected) > selected_points:
        selected = sorted(selected, key=lambda index: (scores[index], index))[:selected_points]

    selected_set = set(selected)
    while len(selected) < min(selected_points, count):
        remaining = np.asarray([index for index in range(count) if index not in selected_set], dtype=np.int32)
        if remaining.size == 0:
            break
        if not selected:
            chosen = min(remaining.tolist(), key=lambda index: (scores[index], index))
        else:
            chosen_points = normalized[np.asarray(selected, dtype=np.int32)]
            delta = normalized[remaining, None, :] - chosen_points[None, :, :]
            minimum_distance_squared = np.min(np.sum(delta * delta, axis=2), axis=1)
            order = sorted(
                range(remaining.size),
                key=lambda offset: (-minimum_distance_squared[offset], scores[remaining[offset]], int(remaining[offset])),
            )
            chosen = int(remaining[order[0]])
        selected.append(chosen)
        selected_set.add(chosen)
    return np.asarray(selected, dtype=np.int32)


def select_top_p_matches(distances: np.ndarray, selected_points: int) -> np.ndarray:
    """Return descriptor-distance Top-P indices for ablation experiments."""

    scores = np.asarray(distances, dtype=np.float64).reshape(-1)
    if selected_points <= 0:
        raise ValueError("selected_points must be positive")
    return np.asarray(sorted(range(scores.size), key=lambda index: (scores[index], index))[:selected_points], dtype=np.int32)


def occupied_grid_cells(
    points: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    *,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> int:
    """Count occupied cells, optionally relative to a shared bounding box."""

    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        return 0
    if bounds is None:
        minimum, maximum = values.min(axis=0), values.max(axis=0)
    else:
        minimum, maximum = (np.asarray(item, dtype=np.float64) for item in bounds)
    span = np.maximum(maximum - minimum, 1e-9)
    normalized = np.clip((values - minimum) / span, 0.0, 1.0)
    columns = np.minimum((normalized[:, 0] * grid_cols).astype(np.int32), grid_cols - 1)
    rows = np.minimum((normalized[:, 1] * grid_rows).astype(np.int32), grid_rows - 1)
    return len(set(zip(rows.tolist(), columns.tolist(), strict=True)))
