"""Initialization for the backbone algorithm."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from .geometry import (
    average_directions,
    estimate_local_similarity,
    project_to_similarity,
    replace_similarity_linear_preserve_center,
    similarity_components,
    similarity_linear_from_scale_direction,
    transform_points_affine,
    wrap_angle,
)
from .io import (
    to_jsonable_matrix,
)
from .models import (
    ImageRecord,
    PairMatch_Edge,
)


def get_edge_by_name(
    edges: Sequence[PairMatch_Edge],
    edge_name: str,
) -> Optional[PairMatch_Edge]:
    """Find an edge by its semantic name in the fixed block graph."""
    return next((edge for edge in edges if edge.edge_name == edge_name), None)


def build_local_similarity_cache(
    edges: Sequence[PairMatch_Edge],
) -> Dict[str, np.ndarray]:
    """Precompute local similarities for valid edges for reuse during initialization."""

    cache: Dict[str, np.ndarray] = {}

    for edge in edges:
        local = estimate_local_similarity(edge)

        if local is None and edge.H_i_to_j is not None:
            local = edge.H_i_to_j

        if local is None:
            continue

        cache[edge.edge_name] = project_to_similarity(local)

    return cache


def propagate_similarity_across_edge(
    known_transform: np.ndarray,
    edge: PairMatch_Edge,
    local_i_to_j: np.ndarray,
    known_idx: int,
    unknown_idx: int,
) -> np.ndarray:
    """Propagate a known global transform: p_j ~= H_i_to_j p_i implies T_j ~= T_i inv(H_i_to_j), or

    T_i ~= T_j H_i_to_j.
    """

    if known_idx == edge.i and unknown_idx == edge.j:
        try:
            propagated = known_transform @ np.linalg.inv(local_i_to_j)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"Cannot invert local similarity for edge {edge.edge_name}") from exc

    elif known_idx == edge.j and unknown_idx == edge.i:
        propagated = known_transform @ local_i_to_j

    else:
        raise ValueError("known_idx / unknown_idx do not match the supplied edge.")

    return project_to_similarity(propagated)


def commonize_stereo_pair_similarity(
    transforms: List[np.ndarray],
    images: Sequence[ImageRecord],
    i: int,
    j: int,
    lock_i: bool,
) -> Dict[str, object]:
    """Commonize stereo linear parts. With lock_i, preserve image i and assign its scale and

    direction to j. Otherwise use geometric-mean scale and mean direction while preserving each
    mapped image center.
    """

    H_i_before = project_to_similarity(transforms[i])
    H_j_before = project_to_similarity(transforms[j])

    scale_i, angle_i, direction_i = similarity_components(H_i_before)
    scale_j, angle_j, direction_j = similarity_components(H_j_before)

    if lock_i:
        common_scale = scale_i
        common_direction = direction_i.copy()
    else:
        common_scale = math.sqrt(max(scale_i * scale_j, config.PROJECTIVE_REG_EPS))
        common_direction = average_directions(
            direction_i,
            direction_j,
        )

    common_linear = similarity_linear_from_scale_direction(
        common_scale,
        common_direction,
    )

    if lock_i:
        transforms[i] = H_i_before
        transforms[j] = replace_similarity_linear_preserve_center(
            H_j_before,
            images[j],
            common_linear,
        )
    else:
        transforms[i] = replace_similarity_linear_preserve_center(
            H_i_before,
            images[i],
            common_linear,
        )
        transforms[j] = replace_similarity_linear_preserve_center(
            H_j_before,
            images[j],
            common_linear,
        )

    scale_i_after, angle_i_after, _ = similarity_components(transforms[i])
    scale_j_after, angle_j_after, _ = similarity_components(transforms[j])

    return {
        "action": "commonize_pair_similarity",
        "i": i,
        "j": j,
        "lock_i": lock_i,
        "scale_i_before": scale_i,
        "scale_j_before": scale_j,
        "scale_ratio_before": scale_i / max(scale_j, config.PROJECTIVE_REG_EPS),
        "angle_i_before_deg": math.degrees(angle_i),
        "angle_j_before_deg": math.degrees(angle_j),
        "angle_difference_before_deg": math.degrees(wrap_angle(angle_i - angle_j)),
        "common_scale": common_scale,
        "common_direction_x": float(common_direction[0]),
        "common_direction_y": float(common_direction[1]),
        "common_angle_deg": math.degrees(math.atan2(common_direction[1], common_direction[0])),
        "scale_i_after": scale_i_after,
        "scale_j_after": scale_j_after,
        "scale_ratio_after": scale_i_after
        / max(
            scale_j_after,
            config.PROJECTIVE_REG_EPS,
        ),
        "angle_i_after_deg": math.degrees(angle_i_after),
        "angle_j_after_deg": math.degrees(angle_j_after),
        "angle_difference_after_deg": math.degrees(wrap_angle(angle_i_after - angle_j_after)),
    }


def affine_stereo_pair_frame(
    transforms: Sequence[np.ndarray],
    i: int,
    j: int,
) -> Tuple[
    float,
    float,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
]:
    """Return the current stereo frame: scales, angles, log scale ratio, shared direction u, and

    normal n.
    """

    scale_i, angle_i, direction_i = similarity_components(transforms[i])
    scale_j, angle_j, direction_j = similarity_components(transforms[j])

    common_direction = average_directions(
        direction_i,
        direction_j,
    )
    normal_direction = np.array(
        [
            -common_direction[1],
            common_direction[0],
        ],
        dtype=np.float64,
    )

    log_scale_ratio = math.log(
        max(scale_i, config.PROJECTIVE_REG_EPS) / max(scale_j, config.PROJECTIVE_REG_EPS)
    )

    return (
        scale_i,
        scale_j,
        angle_i,
        angle_j,
        log_scale_ratio,
        common_direction,
        normal_direction,
    )


def correct_stereo_perpendicular_translation(
    transforms: List[np.ndarray],
    edge: Optional[PairMatch_Edge],
    lock_i: bool,
) -> Dict[str, object]:
    """Correct translation only along the normal to the current shared stereo direction; leave

    parallel displacement unconstrained.
    """

    if edge is None or edge.selected_matches <= 0:
        return {
            "action": "perpendicular_translation_correction",
            "status": "skipped_no_valid_stereo_edge",
        }

    i, j = edge.i, edge.j

    (
        _,
        _,
        angle_i,
        angle_j,
        _,
        common_direction,
        normal_direction,
    ) = affine_stereo_pair_frame(
        transforms,
        i,
        j,
    )

    pi = transform_points_affine(
        transforms[i],
        edge.selected_pts_i,
    )
    pj = transform_points_affine(
        transforms[j],
        edge.selected_pts_j,
    )

    perp_before = (pi - pj) @ normal_direction

    correction = float(np.median(perp_before))

    if lock_i:
        # r_perp = n^T (p_i - p_j)
        # t_j += correction * n  =>  r_perp_new = r_perp - correction
        transforms[j][:2, 2] += correction * normal_direction
    else:
        # Correct non-reference pairs symmetrically to reduce unilateral jumps.
        transforms[i][:2, 2] -= 0.5 * correction * normal_direction
        transforms[j][:2, 2] += 0.5 * correction * normal_direction

    pi_after = transform_points_affine(
        transforms[i],
        edge.selected_pts_i,
    )
    pj_after = transform_points_affine(
        transforms[j],
        edge.selected_pts_j,
    )
    perp_after = (pi_after - pj_after) @ normal_direction

    return {
        "action": "perpendicular_translation_correction",
        "status": "applied",
        "edge_name": edge.edge_name,
        "i": i,
        "j": j,
        "lock_i": lock_i,
        "angle_i_deg": math.degrees(angle_i),
        "angle_j_deg": math.degrees(angle_j),
        "common_direction_x": float(common_direction[0]),
        "common_direction_y": float(common_direction[1]),
        "normal_direction_x": float(normal_direction[0]),
        "normal_direction_y": float(normal_direction[1]),
        "median_perp_before_px": float(np.median(perp_before)),
        "mean_perp_before_px": float(np.mean(perp_before)),
        "std_perp_before_px": float(np.std(perp_before)),
        "applied_correction_px": correction,
        "median_perp_after_px": float(np.median(perp_after)),
        "mean_perp_after_px": float(np.mean(perp_after)),
        "std_perp_after_px": float(np.std(perp_after)),
    }


def initialize_affines(
    num_images: int,
    edges: Sequence[PairMatch_Edge],
    images: Sequence[ImageRecord],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Initialize [left_t, left_t1, right_t, right_t1]. Fix image 0 as the gauge; commonize the

    reference pair, correct its normal translation, propagate ordinary temporal edges first,
    fill missing nodes with other ordinary edges, and allow stereo bridges only as an
    initialization fallback. Commonize the next pair and correct its normal translation
    symmetrically.
    """

    if num_images != 4 or len(images) != 4:
        raise ValueError(
            "Rigid-stereo initialization requires exactly four images "
            "[left_t, left_t+1, right_t, right_t+1]."
        )

    print("Initializing global similarity transforms with rigid stereo-pair prior")

    transforms: List[np.ndarray] = [np.eye(3, dtype=np.float64) for _ in range(num_images)]
    initialized = [False for _ in range(num_images)]

    debug: Dict[str, object] = {
        "method": "rigid_stereo_common_similarity_initialization",
        "image_order": [
            "left_t",
            "left_t+1",
            "right_t",
            "right_t+1",
        ],
        "events": [],
    }
    events = debug["events"]

    local_cache = build_local_similarity_cache(edges)
    edge_by_name = {edge.edge_name: edge for edge in edges}

    # ------------------------------------------------------------
    # Step 1: initialize the reference pair (0, 2) with identity similarities.
    # ------------------------------------------------------------
    transforms[0] = np.eye(3, dtype=np.float64)
    transforms[2] = np.eye(3, dtype=np.float64)
    initialized[0] = True
    initialized[2] = True

    events.append(
        {
            "action": "seed_reference_pair",
            "pair": [0, 2],
            "description": (
                "T0=I; T2 starts with the same rotation/scale. Stereo translation remains free."
            ),
        }
    )

    # Adjust T2 only along the shared normal; keep the T0 gauge fixed.
    stereo_t = edge_by_name.get("stereo_t")
    events.append(
        correct_stereo_perpendicular_translation(
            transforms,
            stereo_t,
            lock_i=True,
        )
    )

    # ------------------------------------------------------------
    # Step 2: prioritize temporal ordinary edges for propagation.
    # ------------------------------------------------------------
    def try_direct_propagation(
        edge_name: str,
        known_idx: int,
        unknown_idx: int,
        reason: str,
    ) -> bool:

        edge = edge_by_name.get(edge_name)
        local = local_cache.get(edge_name)

        if edge is None or local is None:
            events.append(
                {
                    "action": "propagate",
                    "edge_name": edge_name,
                    "status": "skipped_missing_edge_or_local_similarity",
                    "reason": reason,
                }
            )
            return False

        if not initialized[known_idx] or initialized[unknown_idx]:
            return False

        transforms[unknown_idx] = propagate_similarity_across_edge(
            transforms[known_idx],
            edge,
            local,
            known_idx,
            unknown_idx,
        )
        initialized[unknown_idx] = True

        events.append(
            {
                "action": "propagate",
                "edge_name": edge_name,
                "status": "applied",
                "known_idx": known_idx,
                "unknown_idx": unknown_idx,
                "reason": reason,
                "result_transform": to_jsonable_matrix(transforms[unknown_idx]),
            }
        )
        return True

    try_direct_propagation(
        "left_temporal",
        0,
        1,
        "preferred_left_temporal_motion",
    )
    try_direct_propagation(
        "right_temporal",
        2,
        3,
        "preferred_right_temporal_motion",
    )

    # ------------------------------------------------------------
    # Step 3: use other ordinary edges to initialize remaining nodes.
    # ------------------------------------------------------------
    ordinary_edges = [
        edge for edge in edges if edge.edge_type == "ordinary" and edge.edge_name in local_cache
    ]

    progress = True
    while progress and not all(initialized):
        progress = False

        for edge in ordinary_edges:
            local = local_cache[edge.edge_name]

            if initialized[edge.i] and not initialized[edge.j]:
                transforms[edge.j] = propagate_similarity_across_edge(
                    transforms[edge.i],
                    edge,
                    local,
                    edge.i,
                    edge.j,
                )
                initialized[edge.j] = True
                progress = True

                events.append(
                    {
                        "action": "propagate",
                        "edge_name": edge.edge_name,
                        "status": "applied",
                        "known_idx": edge.i,
                        "unknown_idx": edge.j,
                        "reason": "ordinary_fallback_propagation",
                        "result_transform": to_jsonable_matrix(transforms[edge.j]),
                    }
                )

            elif initialized[edge.j] and not initialized[edge.i]:
                transforms[edge.i] = propagate_similarity_across_edge(
                    transforms[edge.j],
                    edge,
                    local,
                    edge.j,
                    edge.i,
                )
                initialized[edge.i] = True
                progress = True

                events.append(
                    {
                        "action": "propagate",
                        "edge_name": edge.edge_name,
                        "status": "applied",
                        "known_idx": edge.j,
                        "unknown_idx": edge.i,
                        "reason": "ordinary_fallback_propagation",
                        "result_transform": to_jsonable_matrix(transforms[edge.i]),
                    }
                )

    # ------------------------------------------------------------
    # Step 4: allow other valid edges only when ordinary connectivity is insufficient.
    # This fallback affects initialization only; the objective retains typed residuals.
    # ------------------------------------------------------------
    if not all(initialized):
        all_cached_edges = [edge for edge in edges if edge.edge_name in local_cache]

        progress = True
        while progress and not all(initialized):
            progress = False

            for edge in all_cached_edges:
                local = local_cache[edge.edge_name]

                if initialized[edge.i] and not initialized[edge.j]:
                    transforms[edge.j] = propagate_similarity_across_edge(
                        transforms[edge.i],
                        edge,
                        local,
                        edge.i,
                        edge.j,
                    )
                    initialized[edge.j] = True
                    progress = True

                    events.append(
                        {
                            "action": "propagate",
                            "edge_name": edge.edge_name,
                            "status": "applied",
                            "known_idx": edge.i,
                            "unknown_idx": edge.j,
                            "reason": "last_resort_any_edge_bridge",
                            "result_transform": to_jsonable_matrix(transforms[edge.j]),
                        }
                    )

                elif initialized[edge.j] and not initialized[edge.i]:
                    transforms[edge.i] = propagate_similarity_across_edge(
                        transforms[edge.j],
                        edge,
                        local,
                        edge.j,
                        edge.i,
                    )
                    initialized[edge.i] = True
                    progress = True

                    events.append(
                        {
                            "action": "propagate",
                            "edge_name": edge.edge_name,
                            "status": "applied",
                            "known_idx": edge.j,
                            "unknown_idx": edge.i,
                            "reason": "last_resort_any_edge_bridge",
                            "result_transform": to_jsonable_matrix(transforms[edge.i]),
                        }
                    )

    for idx, ok in enumerate(initialized):
        if not ok:
            print(
                f"  Warning: image {idx} could not be initialized; "
                "use identity as a numerical fallback."
            )
            transforms[idx] = np.eye(3, dtype=np.float64)

            events.append(
                {
                    "action": "identity_fallback",
                    "image_idx": idx,
                }
            )

    # ------------------------------------------------------------
    # Step 5: commonize the rotation and scale of the next pair (1, 3).
    # ------------------------------------------------------------
    events.append(
        commonize_stereo_pair_similarity(
            transforms,
            images,
            1,
            3,
            lock_i=False,
        )
    )

    # ------------------------------------------------------------
    # Step 6: correct the next pair symmetrically along its shared normal.
    # ------------------------------------------------------------
    stereo_t1 = edge_by_name.get("stereo_t1")
    events.append(
        correct_stereo_perpendicular_translation(
            transforms,
            stereo_t1,
            lock_i=False,
        )
    )

    # Image 0 always remains the fixed gauge.
    transforms[0] = np.eye(3, dtype=np.float64)

    initial_transforms = np.stack(
        [project_to_similarity(H) for H in transforms],
        axis=0,
    )
    initial_transforms[0] = np.eye(3, dtype=np.float64)

    debug["initialized_flags"] = [bool(value) for value in initialized]
    debug["final_initial_transforms"] = [to_jsonable_matrix(H) for H in initial_transforms]

    return initial_transforms, debug
