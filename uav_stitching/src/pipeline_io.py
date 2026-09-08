"""Shared cache loading, connected subsets, and transform persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .optimization_data import MatchObservation
from .utils import ensure_parent


def load_pairs(path: str | Path) -> list[tuple[int, int]]:
    """Load normalized image pairs from Phase 3 JSON."""

    with Path(path).open("r", encoding="utf-8") as stream:
        return [tuple(map(int, pair)) for pair in json.load(stream)["pairs"]]


def select_connected_subset(
    all_image_ids: list[int],
    observations: list[MatchObservation],
    reference_image: int,
    subset_size: int | None,
) -> list[int]:
    """Select a deterministic strongest-edge BFS subset around the reference."""

    if subset_size is None or subset_size >= len(all_image_ids):
        return sorted(all_image_ids)
    if subset_size < 2:
        raise ValueError("subset_size must be at least 2")
    selected = {reference_image}
    while len(selected) < subset_size:
        candidates = []
        for item in observations:
            if (item.image_i in selected) ^ (item.image_j in selected):
                new_id = item.image_j if item.image_i in selected else item.image_i
                candidates.append((-item.inlier_count, item.pairwise_rmse_px, new_id))
        if not candidates:
            raise ValueError("Could not grow a connected subset from the reference image")
        selected.add(min(candidates)[2])
    return sorted(selected)


def save_transform_array(path: str | Path, transforms: dict[int, np.ndarray], total_images: int) -> None:
    """Atomically save an ``(N,3,3)`` array; absent subset images are NaN."""

    destination = Path(path)
    ensure_parent(destination)
    array = np.full((total_images, 3, 3), np.nan, dtype=np.float64)
    for image_id, matrix in transforms.items():
        array[image_id] = matrix
    handle, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.save(stream, array)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_transform_array(path: str | Path) -> dict[int, np.ndarray]:
    """Load finite transforms from an array cache."""

    array = np.load(Path(path), allow_pickle=False)
    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError(f"Unexpected transform array shape: {array.shape}")
    return {index: matrix.astype(np.float64) for index, matrix in enumerate(array) if np.all(np.isfinite(matrix))}
