"""Similarity for the backbone algorithm."""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
from scipy.optimize import least_squares

from . import config
from .geometry import (
    pack_affine_params,
    transform_points_affine,
    unpack_affine_params,
    wrap_angle,
)
from .initialization import (
    affine_stereo_pair_frame,
)
from .models import (
    PairMatch_Edge,
    split_edges_by_type,
)


def affine_ordinary_residuals(
    transforms: np.ndarray,
    ordinary_edges: Sequence[PairMatch_Edge],
) -> np.ndarray:
    """Stage-one ordinary residuals enforce full 2D point consistency."""

    ordinary_count = sum(edge.selected_matches for edge in ordinary_edges)
    if ordinary_count <= 0:
        return np.empty(0, dtype=np.float64)

    ordinary_scale = math.sqrt(float(ordinary_count))
    residual_chunks: List[np.ndarray] = []

    for edge in ordinary_edges:
        pi = transform_points_affine(
            transforms[edge.i],
            edge.selected_pts_i,
        )
        pj = transform_points_affine(
            transforms[edge.j],
            edge.selected_pts_j,
        )
        residual_chunks.append((pi - pj).reshape(-1) / ordinary_scale)

    return np.concatenate(residual_chunks)


def affine_stereo_residuals(
    transforms: np.ndarray,
    stereo_edges: Sequence[PairMatch_Edge],
    stereo_weight_scale: float,
) -> np.ndarray:
    """Stage-one stereo residuals constrain wrapped relative angle, log scale ratio, and n dot

    (p_i_prime - p_j_prime). The normal n is perpendicular to the mean transformed image-x
    direction. Parallel disparity remains free. Scale residuals by sqrt(alpha) before
    robustification; alpha scales energy only within the quadratic loss region.
    """

    if stereo_weight_scale <= 0.0 or not stereo_edges:
        return np.empty(0, dtype=np.float64)

    stereo_count = sum(edge.selected_matches for edge in stereo_edges)
    if stereo_count <= 0:
        return np.empty(0, dtype=np.float64)

    point_normalization = math.sqrt(float(stereo_count))
    edge_normalization = math.sqrt(float(max(len(stereo_edges), 1)))
    alpha_sqrt = math.sqrt(float(stereo_weight_scale))

    residual_chunks: List[np.ndarray] = []

    for edge in stereo_edges:
        (
            scale_i,
            scale_j,
            angle_i,
            angle_j,
            log_scale_ratio,
            _,
            normal_direction,
        ) = affine_stereo_pair_frame(
            transforms,
            edge.i,
            edge.j,
        )

        pi = transform_points_affine(
            transforms[edge.i],
            edge.selected_pts_i,
        )
        pj = transform_points_affine(
            transforms[edge.j],
            edge.selected_pts_j,
        )

        perpendicular = (pi - pj) @ normal_direction

        # Each stereo correspondence contributes one normal residual.
        residual_chunks.append(
            alpha_sqrt
            * math.sqrt(config.LAMBDA_STEREO_PERP)
            * perpendicular
            / (config.STEREO_PERP_SIGMA_PX * point_normalization)
        )

        # Each stereo edge contributes one relative rotation residual.
        angle_difference = wrap_angle(angle_i - angle_j)
        residual_chunks.append(
            np.array(
                [
                    alpha_sqrt
                    * math.sqrt(config.LAMBDA_STEREO_THETA)
                    * angle_difference
                    / (config.STEREO_THETA_SIGMA_RAD * edge_normalization)
                ],
                dtype=np.float64,
            )
        )

        # Each stereo edge contributes one relative scale residual.
        residual_chunks.append(
            np.array(
                [
                    alpha_sqrt
                    * math.sqrt(config.LAMBDA_STEREO_SCALE)
                    * log_scale_ratio
                    / (config.STEREO_LOG_SCALE_SIGMA * edge_normalization)
                ],
                dtype=np.float64,
            )
        )

        # Keep the same frame quantities available for diagnostic definitions.
        _ = scale_i, scale_j

    return np.concatenate(residual_chunks)


def affine_residuals(
    params: np.ndarray,
    num_images: int,
    edges: Sequence[PairMatch_Edge],
    stereo_weight_scale: float = 1.0,
) -> np.ndarray:
    """Combine ordinary and stereo residuals on the same similarity variables. A zero stereo weight

    gives the ordinary-only warm start.
    """

    transforms = unpack_affine_params(params, num_images)
    ordinary_edges, stereo_edges = split_edges_by_type(edges)

    residual_chunks: List[np.ndarray] = []

    ordinary_residual = affine_ordinary_residuals(
        transforms,
        ordinary_edges,
    )
    if ordinary_residual.size > 0:
        residual_chunks.append(ordinary_residual)

    stereo_residual = affine_stereo_residuals(
        transforms,
        stereo_edges,
        stereo_weight_scale,
    )
    if stereo_residual.size > 0:
        residual_chunks.append(stereo_residual)

    if not residual_chunks:
        return np.empty(0, dtype=np.float64)

    return np.concatenate(residual_chunks)


def optimize_affines(
    initial_transforms: np.ndarray,
    edges: Sequence[PairMatch_Edge],
):
    """Optimize global similarities using staged continuation."""

    print("Optimizing global affine/similarity transforms")

    num_images = len(initial_transforms)
    ordinary_edges, stereo_edges = split_edges_by_type(edges)

    ordinary_count = sum(edge.selected_matches for edge in ordinary_edges)
    stereo_count = sum(edge.selected_matches for edge in stereo_edges)

    print(
        f"  valid ordinary edges={len(ordinary_edges)}, matches={ordinary_count}; "
        f"stereo edges={len(stereo_edges)}, matches={stereo_count}"
    )

    x_initial = pack_affine_params(initial_transforms)

    # Report initial and final costs under the complete alpha=1 objective.
    # This keeps costs comparable across continuation stages.
    r0_full = affine_residuals(
        x_initial,
        num_images,
        edges,
        1.0,
    )
    initial_cost = float(0.5 * np.sum(r0_full * r0_full))

    current_x = x_initial.copy()
    last_result = None

    # ------------------------------------------------------------
    # Step A: ordinary-only warm start.
    # ------------------------------------------------------------
    if ordinary_count > 0:
        print("  [Affine stage A] ordinary-only")
        stage_r0 = affine_residuals(
            current_x,
            num_images,
            edges,
            0.0,
        )
        stage_initial_cost = float(0.5 * np.sum(stage_r0 * stage_r0))

        last_result = least_squares(
            affine_residuals,
            current_x,
            args=(num_images, edges, 0.0),
            loss="huber",
            f_scale=4.0,
            max_nfev=config.MAX_OPT_NFEV_AFFINE,
            verbose=0,
        )
        current_x = last_result.x

        stage_r1 = affine_residuals(
            current_x,
            num_images,
            edges,
            0.0,
        )
        stage_final_cost = float(0.5 * np.sum(stage_r1 * stage_r1))
        print(
            f"    cost: {stage_initial_cost:.6f} -> {stage_final_cost:.6f}; "
            f"success={last_result.success}"
        )
    else:
        print("  [Affine stage A] skipped: no valid ordinary matches")

    # ------------------------------------------------------------
    # Step B: retain ordinary terms while increasing stereo weights.
    # ------------------------------------------------------------
    if stereo_count > 0:
        for stage_idx, alpha in enumerate(
            config.AFFINE_STEREO_WEIGHT_SCHEDULE,
            start=1,
        ):
            print(f"  [Affine stereo stage {stage_idx}] ordinary + stereo alpha={alpha:.3f}")

            stage_r0 = affine_residuals(
                current_x,
                num_images,
                edges,
                alpha,
            )
            stage_initial_cost = float(0.5 * np.sum(stage_r0 * stage_r0))

            last_result = least_squares(
                affine_residuals,
                current_x,
                args=(num_images, edges, alpha),
                loss="huber",
                f_scale=4.0,
                max_nfev=config.MAX_OPT_NFEV_AFFINE,
                verbose=0,
            )
            current_x = last_result.x

            stage_r1 = affine_residuals(
                current_x,
                num_images,
                edges,
                alpha,
            )
            stage_final_cost = float(0.5 * np.sum(stage_r1 * stage_r1))
            print(
                f"    cost: {stage_initial_cost:.6f} -> {stage_final_cost:.6f}; "
                f"success={last_result.success}"
            )

    if last_result is None:
        raise RuntimeError("Affine optimization has no valid residuals to optimize.")

    final_transforms = unpack_affine_params(
        current_x,
        num_images,
    )

    r1_full = affine_residuals(
        current_x,
        num_images,
        edges,
        1.0,
    )
    final_cost = float(0.5 * np.sum(r1_full * r1_full))

    return (
        final_transforms,
        last_result,
        initial_cost,
        final_cost,
    )
