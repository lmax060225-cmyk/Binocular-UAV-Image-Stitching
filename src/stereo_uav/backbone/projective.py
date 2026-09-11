"""Projective for the backbone algorithm."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from . import config
from .geometry import (
    compose_similarity_corrections,
    homography_jacobian_general,
    normalize_homography_for_safety,
    normalize_projective_homography,
    pack_variable_projective_params,
    projective_stereo_pair_frame,
    transform_points_projective,
    unpack_variable_projective_params,
    wrap_angle,
)
from .models import (
    ImageRecord,
    PairMatch_Edge,
    split_edges_by_type,
)


def projective_ordinary_residuals(
    transforms: np.ndarray,
    ordinary_edges: Sequence[PairMatch_Edge],
) -> np.ndarray:
    """Stage-two ordinary residuals enforce full 2D point consistency."""

    ordinary_count = sum(edge.selected_matches for edge in ordinary_edges)
    if ordinary_count <= 0:
        return np.empty(0, dtype=np.float64)

    ordinary_scale = math.sqrt(float(ordinary_count))
    residual_chunks: List[np.ndarray] = []

    for edge in ordinary_edges:
        pi = transform_points_projective(
            transforms[edge.i],
            edge.selected_pts_i,
        )
        pj = transform_points_projective(
            transforms[edge.j],
            edge.selected_pts_j,
        )
        residual_chunks.append((pi - pj).reshape(-1) / ordinary_scale)

    return np.concatenate(residual_chunks)


def projective_stereo_residuals(
    transforms: np.ndarray,
    stereo_edges: Sequence[PairMatch_Edge],
    images: Sequence[ImageRecord],
) -> np.ndarray:
    """Stage-two stereo residuals use center-Jacobian local rotations and scales. Match residuals

    constrain only the normal to the shared local x direction; parallel displacement and
    original disparity are not fixed.
    """

    if not stereo_edges:
        return np.empty(0, dtype=np.float64)

    stereo_count = sum(edge.selected_matches for edge in stereo_edges)
    if stereo_count <= 0:
        return np.empty(0, dtype=np.float64)

    point_normalization = math.sqrt(float(stereo_count))
    edge_normalization = math.sqrt(float(max(len(stereo_edges), 1)))

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
        ) = projective_stereo_pair_frame(
            transforms,
            images,
            edge.i,
            edge.j,
        )

        pi = transform_points_projective(
            transforms[edge.i],
            edge.selected_pts_i,
        )
        pj = transform_points_projective(
            transforms[edge.j],
            edge.selected_pts_j,
        )

        perpendicular = (pi - pj) @ normal_direction

        residual_chunks.append(
            math.sqrt(config.LAMBDA_STEREO_PERP)
            * perpendicular
            / (config.STEREO_PERP_SIGMA_PX * point_normalization)
        )

        angle_difference = wrap_angle(angle_i - angle_j)
        residual_chunks.append(
            np.array(
                [
                    math.sqrt(config.LAMBDA_STEREO_THETA)
                    * angle_difference
                    / (config.STEREO_THETA_SIGMA_RAD * edge_normalization)
                ],
                dtype=np.float64,
            )
        )

        residual_chunks.append(
            np.array(
                [
                    math.sqrt(config.LAMBDA_STEREO_SCALE)
                    * log_scale_ratio
                    / (config.STEREO_LOG_SCALE_SIGMA * edge_normalization)
                ],
                dtype=np.float64,
            )
        )

        _ = scale_i, scale_j

    return np.concatenate(residual_chunks)


def projective_normalized_corner_denominators(
    H: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray:
    """Return corner homogeneous denominators normalized relative to the image center."""

    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"H must have shape (3, 3), got {matrix.shape}")

    x_max = float(width) - 1.0
    y_max = float(height) - 1.0
    cx = 0.5 * x_max
    cy = 0.5 * y_max
    h31, h32, h33 = matrix[2]

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        center_denominator = h31 * cx + h32 * cy + h33
        corner_denominators = np.array(
            [
                h33,
                h31 * x_max + h33,
                h32 * y_max + h33,
                h31 * x_max + h32 * y_max + h33,
            ],
            dtype=np.float64,
        )
        normalized = (
            corner_denominators
            * center_denominator
            / (center_denominator * center_denominator + config.PROJECTIVE_REG_EPS)
        )

    return np.nan_to_num(
        np.asarray(normalized, dtype=np.float64),
        nan=-1.0e6,
        posinf=-1.0e6,
        neginf=-1.0e6,
    )


def make_projective_safety_grid() -> np.ndarray:
    """Generate a fixed row-major grid over normalized source coordinates [-1, 1]^2."""

    rows = int(config.PROJECTIVE_SAFETY_GRID_ROWS)
    cols = int(config.PROJECTIVE_SAFETY_GRID_COLS)
    if rows <= 0 or cols <= 0:
        raise ValueError(f"PROJECTIVE_SAFETY_GRID_ROWS/COLS must be positive, got {rows}, {cols}.")

    u_values = np.linspace(-1.0, 1.0, cols, dtype=np.float64)
    v_values = np.linspace(-1.0, 1.0, rows, dtype=np.float64)
    return np.asarray(
        [(u, v) for v in v_values for u in u_values],
        dtype=np.float64,
    )


def smooth_positive_barrier(violation: float, softness: float) -> float:
    """Evaluate tau * softplus(violation / tau) stably."""

    tau = float(softness)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"softness must be finite and positive, got {softness}")

    violation = float(violation)
    if math.isnan(violation) or violation == math.inf:
        return 1.0e6
    if violation == -math.inf:
        return 0.0

    residual = tau * float(np.logaddexp(0.0, violation / tau))
    return residual if math.isfinite(residual) else 1.0e6


def projective_native_safety_residual_size(num_regularized: int) -> int:
    """Return the fixed residual dimension for denominator and singular-value barriers."""

    num_regularized = int(num_regularized)
    if num_regularized < 0:
        raise ValueError("num_regularized must be non-negative.")
    grid_size = int(config.PROJECTIVE_SAFETY_GRID_ROWS) * int(config.PROJECTIVE_SAFETY_GRID_COLS)
    return num_regularized * (4 + 2 * grid_size)


def _projective_local_singular_values(
    H_bar: np.ndarray,
    sample_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the minimum and maximum Jacobian singular values over the safety grid."""

    dangerous_min = config.PROJECTIVE_REG_EPS
    dangerous_max = 1.0 / config.PROJECTIVE_REG_EPS
    sigma_mins: List[float] = []
    sigma_maxs: List[float] = []

    for u, v in np.asarray(sample_points, dtype=np.float64):
        J = homography_jacobian_general(H_bar, float(u), float(v))
        if J is None:
            sigma_mins.append(dangerous_min)
            sigma_maxs.append(dangerous_max)
            continue

        try:
            singular_values = np.linalg.svd(J, compute_uv=False)
        except np.linalg.LinAlgError:
            singular_values = np.empty(0, dtype=np.float64)

        if singular_values.shape != (2,) or not np.all(np.isfinite(singular_values)):
            sigma_min = dangerous_min
            sigma_max = dangerous_max
        else:
            sigma_min = max(float(np.min(singular_values)), config.PROJECTIVE_REG_EPS)
            sigma_max = max(float(np.max(singular_values)), sigma_min)

        sigma_mins.append(sigma_min)
        sigma_maxs.append(sigma_max)

    return (
        np.asarray(sigma_mins, dtype=np.float64),
        np.asarray(sigma_maxs, dtype=np.float64),
    )


def projective_native_safety_residuals(
    corrections: np.ndarray,
    images: Sequence[ImageRecord],
    regularized_indices: Sequence[int],
) -> np.ndarray:
    """Apply denominator and singular-value barriers directly to local corrections C_i in G_i = S_i

    @ C_i. Evaluate each correction in its own image-normalized coordinates; the reference
    transform cannot absorb a common projective drift from this penalty.
    """

    corrections = np.asarray(corrections, dtype=np.float64)
    if corrections.ndim != 3 or corrections.shape[1:] != (3, 3):
        raise ValueError("corrections must have shape (N, 3, 3).")
    if len(images) != len(corrections):
        raise ValueError("images and corrections must have equal length.")

    indices = [int(idx) for idx in regularized_indices]
    if len(indices) != len(set(indices)):
        raise ValueError("regularized_indices contains duplicate indices.")
    for idx in indices:
        if idx < 0 or idx >= len(corrections):
            raise IndexError(f"Regularized index {idx} is outside corrections.")
    if not indices:
        return np.empty(0, dtype=np.float64)

    lambda_denom = float(config.LAMBDA_PROJECTIVE_DENOM)
    lambda_sv = float(config.LAMBDA_PROJECTIVE_SV)
    for name, value in (
        ("LAMBDA_PROJECTIVE_DENOM", lambda_denom),
        ("LAMBDA_PROJECTIVE_SV", lambda_sv),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}.")

    sv_min = float(config.PROJECTIVE_SV_MIN)
    sv_max = float(config.PROJECTIVE_SV_MAX)
    if not math.isfinite(sv_min) or not math.isfinite(sv_max) or sv_min <= 0.0 or sv_min >= sv_max:
        raise ValueError("Projective singular-value limits must satisfy 0 < min < max.")

    sample_points = make_projective_safety_grid()
    num_regularized = len(indices)
    grid_size = len(sample_points)

    denom_scale = math.sqrt(lambda_denom / float(max(4 * num_regularized, 1)))
    sv_scale = math.sqrt(lambda_sv / float(max(2 * grid_size * num_regularized, 1)))

    log_sv_min = math.log(sv_min)
    log_sv_max = math.log(sv_max)

    denominator_residuals: List[float] = []
    singular_value_residuals: List[float] = []

    for idx in indices:
        correction = normalize_projective_homography(corrections[idx])

        normalized_denominators = projective_normalized_corner_denominators(
            correction,
            images[idx].width,
            images[idx].height,
        )
        denominator_residuals.extend(
            denom_scale
            * smooth_positive_barrier(
                config.PROJECTIVE_DENOM_MIN - float(value),
                config.PROJECTIVE_DENOM_SOFTNESS,
            )
            for value in normalized_denominators
        )

        # Use the image's own pixel scale on both sides of C_i.
        correction_bar = normalize_homography_for_safety(
            correction,
            images[idx],
            images[idx],
        )
        sigma_mins, sigma_maxs = _projective_local_singular_values(
            correction_bar,
            sample_points,
        )

        for sigma_min, sigma_max in zip(sigma_mins, sigma_maxs):
            log_sigma_min = math.log(max(float(sigma_min), config.PROJECTIVE_REG_EPS))
            log_sigma_max = math.log(max(float(sigma_max), config.PROJECTIVE_REG_EPS))
            singular_value_residuals.extend(
                [
                    sv_scale
                    * smooth_positive_barrier(
                        log_sv_min - log_sigma_min,
                        config.PROJECTIVE_SV_SOFTNESS,
                    ),
                    sv_scale
                    * smooth_positive_barrier(
                        log_sigma_max - log_sv_max,
                        config.PROJECTIVE_SV_SOFTNESS,
                    ),
                ]
            )

    result = np.asarray(
        denominator_residuals + singular_value_residuals,
        dtype=np.float64,
    )
    expected_size = projective_native_safety_residual_size(num_regularized)
    if result.size != expected_size:
        raise RuntimeError(
            "Native projective safety residual dimension changed unexpectedly: "
            f"{result.size} != {expected_size}."
        )
    return np.nan_to_num(
        result,
        nan=1.0e6,
        posinf=1.0e6,
        neginf=1.0e6,
    )


def projective_data_residuals(
    transforms: np.ndarray,
    edges: Sequence[PairMatch_Edge],
    images: Sequence[ImageRecord],
) -> np.ndarray:
    """Evaluate ordinary and stereo data terms with their respective residual definitions."""

    ordinary_edges, stereo_edges = split_edges_by_type(edges)
    chunks: List[np.ndarray] = []

    ordinary_residual = projective_ordinary_residuals(transforms, ordinary_edges)
    if ordinary_residual.size > 0:
        chunks.append(ordinary_residual)

    stereo_residual = projective_stereo_residuals(transforms, stereo_edges, images)
    if stereo_residual.size > 0:
        chunks.append(stereo_residual)

    return (
        np.concatenate(chunks).astype(np.float64, copy=False)
        if chunks
        else np.empty(0, dtype=np.float64)
    )


def projective_objective_residuals(
    global_transforms: np.ndarray,
    corrections: np.ndarray,
    edges: Sequence[PairMatch_Edge],
    images: Sequence[ImageRecord],
    regularized_indices: Sequence[int],
) -> np.ndarray:
    """Evaluate typed data terms on G_i and native safety terms only on C_i."""

    chunks: List[np.ndarray] = []

    data_residual = projective_data_residuals(global_transforms, edges, images)
    if data_residual.size > 0:
        chunks.append(data_residual)

    safety_residual = projective_native_safety_residuals(
        corrections,
        images,
        regularized_indices,
    )
    if safety_residual.size > 0:
        chunks.append(safety_residual)

    return (
        np.concatenate(chunks).astype(np.float64, copy=False)
        if chunks
        else np.empty(0, dtype=np.float64)
    )


def projective_objective_expected_residual_size(
    edges: Sequence[PairMatch_Edge],
    num_regularized: int,
) -> int:
    """Return the fixed combined residual dimension for typed observations and safety barriers."""

    data_size = 0
    for edge in edges:
        if edge.edge_type == "ordinary":
            data_size += 2 * int(edge.selected_matches)
        elif edge.edge_type == "stereo":
            data_size += int(edge.selected_matches) + 2
        else:
            raise ValueError(f"Unknown edge_type={edge.edge_type!r}")

    return data_size + projective_native_safety_residual_size(num_regularized)


def backbone_correction_projective_residuals(
    params: np.ndarray,
    backbones: np.ndarray,
    base_corrections: np.ndarray,
    variable_indices: Sequence[int],
    edges: Sequence[PairMatch_Edge],
    images: Sequence[ImageRecord],
) -> np.ndarray:
    """Shared local and persistent projective objective. Optimize selected C_i, compose G_i = S_i @

    C_i, evaluate data terms on G_i, and safety terms on C_i.
    """

    expected_size = projective_objective_expected_residual_size(
        edges,
        len(variable_indices),
    )
    if expected_size <= 0:
        return np.empty(0, dtype=np.float64)

    try:
        corrections = unpack_variable_projective_params(
            params,
            base_corrections,
            variable_indices,
        )
        global_transforms = compose_similarity_corrections(
            backbones,
            corrections,
        )
        residual = projective_objective_residuals(
            global_transforms=global_transforms,
            corrections=corrections,
            edges=edges,
            images=images,
            regularized_indices=variable_indices,
        )
        if residual.size != expected_size:
            raise RuntimeError(
                "Projective residual dimension changed unexpectedly: "
                f"{residual.size} != {expected_size}."
            )
        if not np.all(np.isfinite(residual)):
            raise FloatingPointError("Projective residual contains NaN or Inf.")
        return residual

    except (
        ValueError,
        IndexError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
        OverflowError,
    ):
        return np.full(
            expected_size,
            config.PERSISTENT_INVALID_RESIDUAL,
            dtype=np.float64,
        )


def optimize_projectives(
    initial_affines: np.ndarray,
    edges: Sequence[PairMatch_Edge],
    images: Sequence[ImageRecord],
):
    """Block-local stage-two optimization. Fix the stage-one similarities S_i and optimize only

    C_i, with typed observations on H_i = S_i @ C_i and native safety on C_i.
    """

    print("Optimizing block-local projective corrections on similarity backbones")
    backbones = np.asarray(initial_affines, dtype=np.float64).copy()
    num_images = len(backbones)
    if len(images) != num_images:
        raise ValueError(
            "The number of images must match the number of initial affine transforms."
            f"{len(images)} != {num_images}"
        )

    backbones = np.stack(
        [normalize_projective_homography(S) for S in backbones],
        axis=0,
    )
    # Remove numerical bottom-row drift from the stage-one similarity.
    backbones[:, 2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    ordinary_edges, stereo_edges = split_edges_by_type(edges)
    print(
        f"  projective objective: ordinary_edges={len(ordinary_edges)}, "
        f"stereo_edges={len(stereo_edges)}, correction_safety=denominator+singular_values"
    )

    base_corrections = np.repeat(
        np.eye(3, dtype=np.float64)[None, :, :],
        num_images,
        axis=0,
    )
    variable_indices = list(range(1, num_images))
    x0 = pack_variable_projective_params(base_corrections, variable_indices)
    residual_args = (
        backbones,
        base_corrections,
        variable_indices,
        edges,
        images,
    )

    r0 = backbone_correction_projective_residuals(x0, *residual_args)
    initial_cost = float(0.5 * np.sum(r0 * r0))

    result = least_squares(
        backbone_correction_projective_residuals,
        x0,
        args=residual_args,
        loss="soft_l1",
        f_scale=4.0,
        max_nfev=config.MAX_OPT_NFEV_PROJECTIVE,
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
    return final_transforms, result, initial_cost, final_cost
