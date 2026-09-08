"""Global projective optimization with the paper's rigid regularizer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .optimization_data import MatchObservation, assert_observation_graph_connected
from .transforms import decode_homography_parameters, encode_homography_parameters


@dataclass(frozen=True)
class ProjectiveOptimizationResult:
    transforms: dict[int, np.ndarray]
    reference_image: int
    initial_cost: float
    final_cost: float
    initial_match_rmse_px: float
    final_match_rmse_px: float
    final_match_energy_average: float
    final_rigid_energy_average: float
    nfev: int
    optimality: float
    status: int
    success: bool
    message: str


def projective_matching_residuals(
    transforms: dict[int, np.ndarray],
    observations: list[MatchObservation],
    *,
    scale: float = 1.0,
    denominator_epsilon: float = 1e-8,
) -> np.ndarray:
    """Return selected-point projective residuals in global pixels.

    Every ``H_i`` maps image pixel ``(x,y)`` to global mosaic coordinates.
    Homogeneous coordinates are divided by the third component. For each
    selected correspondence, the residual is ``scale*(warp(H_i,p_i)-
    warp(H_j,p_j))`` ordered ``rx,ry``. Near-zero denominators are clipped only
    to keep numerical evaluation finite during trust-region trial steps.
    """

    chunks = []
    for item in observations:
        first = _warp_projective_safe(transforms[item.image_i], item.selected_pts_i, denominator_epsilon)
        second = _warp_projective_safe(transforms[item.image_j], item.selected_pts_j, denominator_epsilon)
        chunks.append(((first - second) * scale).reshape(-1))
    return np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.float64)


def rigid_residuals(
    transforms: dict[int, np.ndarray],
    image_ids: list[int],
    reference_image: int,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Return paper rigid residuals for each non-reference homography.

    For ``H=[[a,b,c],[d,e,f],[g,h,1]]`` the four residuals are
    ``a*b+d*e``, ``a^2+d^2-1``, ``b^2+e^2-1``, and ``g^2+h^2``. Multiplying by
    ``sqrt(omega/N)`` makes their squared sum equal the weighted average rigid
    energy in Eq. (19), apart from SciPy's conventional factor one-half.
    """

    values = []
    for image_id in image_ids:
        if image_id == reference_image:
            continue
        matrix = transforms[image_id]
        a, b, d, e, g, h = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1], matrix[2, 0], matrix[2, 1]
        values.extend((a * b + d * e, a * a + d * d - 1.0, b * b + e * e - 1.0, g * g + h * h))
    return np.asarray(values, dtype=np.float64) * scale


def projective_residuals(
    parameters: np.ndarray,
    image_ids: list[int],
    reference_image: int,
    observations: list[MatchObservation],
    sigma_tr: float,
    rigid_weight: float,
) -> np.ndarray:
    """Return residuals equivalent to averaged paper objective Eq. (19).

    Matching residuals use ``1/sqrt(total selected point pairs)`` so their
    squared sum is average projective matching energy. Rigid residuals use
    ``sqrt(rigid_weight/N)``. Parameters encode translations as
    ``c/sigma_tr,f/sigma_tr``; all decoded matrices map image pixels to global
    mosaic coordinates and the reference image is fixed to identity.
    """

    transforms = _decode_projective_state(parameters, image_ids, reference_image, sigma_tr)
    point_count = sum(item.selected_pts_i.shape[0] for item in observations)
    if point_count == 0:
        raise ValueError("Projective optimization requires selected matches")
    matching = projective_matching_residuals(transforms, observations, scale=1.0 / np.sqrt(point_count))
    rigid = rigid_residuals(
        transforms, image_ids, reference_image,
        scale=np.sqrt(rigid_weight / max(1, len(image_ids))),
    )
    return np.concatenate((matching, rigid))


def optimize_projectives(
    image_ids: list[int],
    observations: list[MatchObservation],
    reference_image: int,
    affine_transforms: dict[int, np.ndarray],
    *,
    sigma_tr: float = 5000.0,
    rigid_weight: float = 800.0,
    max_nfev: int = 300,
    verbose: int = 0,
) -> ProjectiveOptimizationResult:
    """Optimize all non-reference homographies from affine initialization."""

    assert_observation_graph_connected(image_ids, observations)
    if rigid_weight < 0:
        raise ValueError("rigid_weight must be non-negative")
    variable_ids = [image_id for image_id in image_ids if image_id != reference_image]
    x0 = np.concatenate([encode_homography_parameters(affine_transforms[image_id], sigma_tr) for image_id in variable_ids])
    raw_initial_matching = projective_matching_residuals(affine_transforms, observations)
    initial_combined = projective_residuals(x0, image_ids, reference_image, observations, sigma_tr, rigid_weight)
    sparsity = _projective_jacobian_sparsity(image_ids, reference_image, observations)
    result = least_squares(
        projective_residuals,
        x0,
        args=(image_ids, reference_image, observations, sigma_tr, rigid_weight),
        method="trf",
        jac_sparsity=sparsity,
        x_scale="jac",
        max_nfev=max_nfev,
        verbose=verbose,
    )
    transforms = _decode_projective_state(result.x, image_ids, reference_image, sigma_tr)
    raw_final_matching = projective_matching_residuals(transforms, observations)
    raw_rigid = rigid_residuals(transforms, image_ids, reference_image)
    point_count = sum(item.selected_pts_i.shape[0] for item in observations)
    return ProjectiveOptimizationResult(
        transforms=transforms,
        reference_image=reference_image,
        initial_cost=0.5 * float(initial_combined @ initial_combined),
        final_cost=float(result.cost),
        initial_match_rmse_px=_point_rmse(raw_initial_matching),
        final_match_rmse_px=_point_rmse(raw_final_matching),
        final_match_energy_average=float(raw_final_matching @ raw_final_matching) / point_count,
        final_rigid_energy_average=float(raw_rigid @ raw_rigid) / max(1, len(image_ids)),
        nfev=int(result.nfev),
        optimality=float(result.optimality),
        status=int(result.status),
        success=bool(result.success),
        message=str(result.message),
    )


def _decode_projective_state(parameters: np.ndarray, image_ids: list[int], reference_image: int,
                             sigma_tr: float) -> dict[int, np.ndarray]:
    transforms = {reference_image: np.eye(3, dtype=np.float64)}
    offset = 0
    for image_id in image_ids:
        if image_id == reference_image:
            continue
        transforms[image_id] = decode_homography_parameters(parameters[offset:offset + 8], sigma_tr)
        offset += 8
    return transforms


def _warp_projective_safe(transform: np.ndarray, points: np.ndarray, epsilon: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    x, y = values[:, 0], values[:, 1]
    denominator = transform[2, 0] * x + transform[2, 1] * y + 1.0
    signs = np.where(denominator < 0.0, -1.0, 1.0)
    safe = np.where(np.abs(denominator) < epsilon, signs * epsilon, denominator)
    first = (transform[0, 0] * x + transform[0, 1] * y + transform[0, 2]) / safe
    second = (transform[1, 0] * x + transform[1, 1] * y + transform[1, 2]) / safe
    return np.column_stack((first, second))


def _point_rmse(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(residual.reshape(-1, 2) ** 2, axis=1)))) if residual.size else 0.0


def _projective_jacobian_sparsity(image_ids: list[int], reference_image: int,
                                  observations: list[MatchObservation]) -> lil_matrix:
    variable_ids = [image_id for image_id in image_ids if image_id != reference_image]
    columns = {image_id: 8 * index for index, image_id in enumerate(variable_ids)}
    matching_rows = 2 * sum(item.selected_pts_i.shape[0] for item in observations)
    rigid_rows = 4 * len(variable_ids)
    pattern = lil_matrix((matching_rows + rigid_rows, 8 * len(variable_ids)), dtype=np.int8)
    row = 0
    for item in observations:
        rows = 2 * item.selected_pts_i.shape[0]
        for image_id in (item.image_i, item.image_j):
            if image_id in columns:
                pattern[row:row + rows, columns[image_id]:columns[image_id] + 8] = 1
        row += rows
    for image_id in variable_ids:
        pattern[row:row + 4, columns[image_id]:columns[image_id] + 8] = 1
        row += 4
    return pattern
