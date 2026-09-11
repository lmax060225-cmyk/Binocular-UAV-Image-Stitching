"""Persistent for the backbone algorithm."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from . import config
from .geometry import (
    compose_similarity_corrections,
    decompose_global_transforms,
    normalize_projective_homography,
    pack_variable_projective_params,
    project_homography_to_similarity_backbone,
    unpack_variable_projective_params,
)
from .models import (
    ImageRecord,
    PairMatch_Edge,
    PersistentBlockCache,
    split_edges_by_type,
)
from .projective import (
    backbone_correction_projective_residuals,
)


# ==================== PERSISTENT TWO-BLOCK BACKBONE/CORRECTION OPTIMIZATION ====================
def build_persistent_active_window(
    previous_block: PersistentBlockCache,
    current_keys: Sequence[Tuple[str, int]],
    current_images: Sequence[ImageRecord],
    current_edges: Sequence[PairMatch_Edge],
) -> Tuple[
    List[Tuple[str, int]],
    List[ImageRecord],
    List[PairMatch_Edge],
]:
    """Merge consecutive blocks and remap their original typed observations. Preserve shared stereo

    observation multiplicity; no factor deduplication is performed.
    """

    previous_keys = list(previous_block.keys)
    current_keys = list(current_keys)
    if len(previous_keys) != 4 or len(current_keys) != 4:
        raise ValueError("Persistent two-block window requires two 4-image blocks.")

    previous_key_set = set(previous_keys)
    current_key_set = set(current_keys)
    shared_keys = previous_key_set & current_key_set
    if len(shared_keys) != 2:
        raise ValueError(
            "Adjacent blocks must share exactly one stereo pair (2 images), "
            f"but shared {len(shared_keys)} keys: {sorted(shared_keys)}"
        )

    # Preserve previous-block order, then append new current-block nodes.
    active_keys = previous_keys.copy()
    for key in current_keys:
        if key not in active_keys:
            active_keys.append(key)

    if len(active_keys) != 6:
        raise ValueError(
            "Two adjacent 4-image blocks should form exactly 6 unique images, "
            f"got {len(active_keys)}."
        )

    image_by_key: Dict[Tuple[str, int], ImageRecord] = {}
    for key, image in zip(previous_keys, previous_block.images):
        image_by_key[key] = image
    for key, image in zip(current_keys, current_images):
        image_by_key[key] = image

    active_images = [image_by_key[key] for key in active_keys]
    active_index = {key: idx for idx, key in enumerate(active_keys)}

    active_edges: List[PairMatch_Edge] = []

    def add_block_edges(
        block_label: str,
        block_keys: Sequence[Tuple[str, int]],
        edges: Sequence[PairMatch_Edge],
    ) -> None:
        for edge in edges:
            if edge.i < 0 or edge.i >= len(block_keys):
                raise IndexError(f"Invalid edge.i={edge.i} for {block_label} block.")
            if edge.j < 0 or edge.j >= len(block_keys):
                raise IndexError(f"Invalid edge.j={edge.j} for {block_label} block.")

            key_i = block_keys[edge.i]
            key_j = block_keys[edge.j]
            remapped = replace(
                edge,
                i=active_index[key_i],
                j=active_index[key_j],
                edge_name=f"{block_label}:{edge.edge_name}",
            )
            active_edges.append(remapped)

    add_block_edges(
        "previous",
        previous_keys,
        previous_block.valid_edges,
    )
    add_block_edges(
        "current",
        current_keys,
        current_edges,
    )

    return active_keys, active_images, active_edges


def initialize_persistent_active_state(
    active_keys: Sequence[Tuple[str, int]],
    current_keys: Sequence[Tuple[str, int]],
    current_images: Sequence[ImageRecord],
    current_local_projective: np.ndarray,
    global_transforms_by_key: Dict[Tuple[str, int], np.ndarray],
    global_backbones_by_key: Dict[Tuple[str, int], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize a two-block window from persistent S_old and G_old. Propagate new similarities

    using Q_sim = S_shared_old @ inv(S_shared_local). Local corrections initialize G_new but do
    not enter Q_sim.
    """

    current_keys = list(current_keys)
    current_local = np.asarray(current_local_projective, dtype=np.float64)
    if current_local.shape != (len(current_keys), 3, 3):
        raise ValueError("current_local_projective must match current_keys and have shape (N,3,3).")
    if len(current_images) != len(current_keys):
        raise ValueError("current_images and current_keys must have equal length.")

    current_local_backbones, current_local_corrections = decompose_global_transforms(
        current_local,
        current_images,
    )
    current_key_to_local = {key: idx for idx, key in enumerate(current_keys)}

    available_shared_local_indices = [
        idx
        for idx, key in enumerate(current_keys)
        if key in global_transforms_by_key and key in global_backbones_by_key
    ]
    if not available_shared_local_indices:
        raise ValueError(
            "Cannot initialize a connected persistent window without shared S_old/G_old."
        )

    anchor_local_idx = next(
        (idx for idx in (0, 2) if idx in available_shared_local_indices),
        available_shared_local_indices[0],
    )
    anchor_key = current_keys[anchor_local_idx]
    S_anchor_global = normalize_projective_homography(global_backbones_by_key[anchor_key])
    S_anchor_local = normalize_projective_homography(current_local_backbones[anchor_local_idx])
    Q_similarity = normalize_projective_homography(S_anchor_global @ np.linalg.inv(S_anchor_local))
    # Project again to ensure the propagated gauge has no projective bottom row.
    Q_similarity = project_homography_to_similarity_backbone(
        Q_similarity,
        current_images[anchor_local_idx],
    )

    active_backbones: List[np.ndarray] = []
    active_transforms: List[np.ndarray] = []
    for key in active_keys:
        if key in global_transforms_by_key and key in global_backbones_by_key:
            S = normalize_projective_homography(global_backbones_by_key[key])
            G = normalize_projective_homography(global_transforms_by_key[key])
        elif key in current_key_to_local:
            local_idx = current_key_to_local[key]
            S = normalize_projective_homography(Q_similarity @ current_local_backbones[local_idx])
            C_local = current_local_corrections[local_idx]
            G = normalize_projective_homography(S @ C_local)
        else:
            raise KeyError(
                f"Active key {key} has neither persistent state nor current local state."
            )
        active_backbones.append(S)
        active_transforms.append(G)

    return (
        np.stack(active_backbones, axis=0),
        np.stack(active_transforms, axis=0),
    )


def optimize_persistent_window_projectives(
    initial_global_backbones: np.ndarray,
    initial_global_transforms: np.ndarray,
    variable_indices: Sequence[int],
    window_edges: Sequence[PairMatch_Edge],
    window_images: Sequence[ImageRecord],
):
    """Optimize the previous and current blocks in a fixed-lag window. Hold S_i fixed during least

    squares, vary C_i, and fix the oldest pair's corrections to preserve the historical gauge.
    """

    backbones = np.asarray(initial_global_backbones, dtype=np.float64)
    transforms = np.asarray(initial_global_transforms, dtype=np.float64)
    if backbones.shape != transforms.shape or backbones.ndim != 3 or backbones.shape[1:] != (3, 3):
        raise ValueError("initial backbones/transforms must both have shape (N,3,3).")
    if len(transforms) != len(window_images):
        raise ValueError("window_images and transforms must have equal length.")

    backbones = np.stack(
        [normalize_projective_homography(S) for S in backbones],
        axis=0,
    )
    transforms = np.stack(
        [normalize_projective_homography(G) for G in transforms],
        axis=0,
    )
    _, base_corrections = decompose_global_transforms(
        transforms,
        window_images,
        backbone_template=backbones,
    )

    variable_indices = [int(idx) for idx in variable_indices]
    variable_set = set(variable_indices)
    if not variable_indices:
        raise ValueError("Persistent window has no variable corrections.")
    if len(variable_indices) != len(variable_set):
        raise ValueError("variable_indices contains duplicate indices.")

    fixed_indices = [idx for idx in range(len(transforms)) if idx not in variable_set]
    if not fixed_indices:
        raise ValueError("Persistent optimization must keep at least one historical state fixed.")

    influential_edges = [
        edge for edge in window_edges if edge.i in variable_set or edge.j in variable_set
    ]
    if not influential_edges:
        raise ValueError("Persistent window has no factor touching a variable correction.")

    ordinary_edges, stereo_edges = split_edges_by_type(influential_edges)
    print("Optimizing persistent two-block corrections on similarity backbones")
    print(f"  nodes={len(transforms)}, fixed={fixed_indices}, variable={variable_indices}")
    print(
        f"  typed factors: ordinary={len(ordinary_edges)}, "
        f"stereo={len(stereo_edges)}, total={len(influential_edges)}; "
        "correction_safety=denominator+singular_values"
    )

    x0 = pack_variable_projective_params(base_corrections, variable_indices)
    residual_args = (
        backbones,
        base_corrections,
        variable_indices,
        influential_edges,
        window_images,
    )

    r0 = backbone_correction_projective_residuals(x0, *residual_args)
    if r0.size == 0:
        raise RuntimeError("Persistent projective objective has no residuals.")
    initial_cost = float(0.5 * np.sum(r0 * r0))

    result = least_squares(
        backbone_correction_projective_residuals,
        x0,
        args=residual_args,
        loss="soft_l1",
        f_scale=config.PERSISTENT_PROJECTIVE_F_SCALE,
        max_nfev=config.MAX_OPT_NFEV_PERSISTENT_PROJECTIVE,
        verbose=1,
    )

    final_corrections = unpack_variable_projective_params(
        result.x,
        base_corrections,
        variable_indices,
    )
    final_transforms = compose_similarity_corrections(
        backbones,
        final_corrections,
    )
    r1 = backbone_correction_projective_residuals(result.x, *residual_args)
    final_cost = float(0.5 * np.sum(r1 * r1))

    return (
        final_transforms,
        result,
        initial_cost,
        final_cost,
        influential_edges,
    )
