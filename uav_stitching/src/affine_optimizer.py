"""Paper shared-variable global affine registration optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .optimization_data import MatchObservation, assert_observation_graph_connected
from .transforms import decode_affine_parameters, encode_affine_parameters, estimate_similarity, normalize_homography


@dataclass(frozen=True)
class AffineOptimizationResult:
    transforms: dict[int, np.ndarray]
    reference_image: int
    initial_cost: float
    final_cost: float
    initial_rmse_px: float
    final_rmse_px: float
    nfev: int
    optimality: float
    status: int
    success: bool
    message: str


def initialize_affines(
    image_ids: list[int],
    observations: list[MatchObservation],
    reference_image: int,
    sigma_tr: float,
) -> dict[int, np.ndarray]:
    """Propagate robust pairwise similarities over a strongest-edge tree.

    For a cached pair, ``S_ij`` maps image ``i`` pixels to image ``j`` pixels.
    Global transforms map image pixels to the reference mosaic, so when ``i``
    is known: ``H_j = H_i @ inv(S_ij)``; when ``j`` is known:
    ``H_i = H_j @ S_ij``. This propagation strategy is an implementation
    choice because the paper does not specify affine initialization details.
    """

    assert_observation_graph_connected(image_ids, observations)
    if reference_image not in image_ids:
        raise ValueError("Reference image is not in image_ids")
    transforms: dict[int, np.ndarray] = {reference_image: np.eye(3, dtype=np.float64)}
    remaining = set(image_ids) - {reference_image}
    pairwise: dict[tuple[int, int], np.ndarray] = {}
    for item in observations:
        pairwise[(item.image_i, item.image_j)] = estimate_similarity(item.all_pts_i, item.all_pts_j)

    while remaining:
        candidates = []
        for item in observations:
            crosses = (item.image_i in transforms and item.image_j in remaining) or (
                item.image_j in transforms and item.image_i in remaining
            )
            if crosses:
                candidates.append((-item.inlier_count, item.pairwise_rmse_px, item.image_i, item.image_j, item))
        if not candidates:
            raise ValueError(f"Could not initialize disconnected images: {sorted(remaining)}")
        item = min(candidates)[-1]
        similarity = pairwise[(item.image_i, item.image_j)]
        if item.image_i in transforms:
            new_id = item.image_j
            transform = transforms[item.image_i] @ np.linalg.inv(similarity)
        else:
            new_id = item.image_i
            transform = transforms[item.image_j] @ similarity
        transform = normalize_homography(transform)
        transforms[new_id] = decode_affine_parameters(encode_affine_parameters(transform, sigma_tr), sigma_tr)
        remaining.remove(new_id)
    return transforms


def affine_residuals(
    parameters: np.ndarray,
    image_ids: list[int],
    reference_image: int,
    observations: list[MatchObservation],
    sigma_tr: float,
) -> np.ndarray:
    """Return stacked global correspondence residuals for affine optimization.

    Each non-reference image uses ``[a,b,c/sigma_tr,f/sigma_tr]`` and decodes
    to ``H_i=[[a,b,c],[-b,a,f],[0,0,1]]``. Every ``H_i`` maps image pixel
    coordinates to the shared global mosaic. For selected match ``p_i,p_j``,
    residuals are ``H_i p_i - H_j p_j`` in global pixels, ordered ``rx,ry``.
    """

    transforms = _decode_affine_state(parameters, image_ids, reference_image, sigma_tr)
    chunks = []
    for item in observations:
        first = _warp_affine(transforms[item.image_i], item.selected_pts_i)
        second = _warp_affine(transforms[item.image_j], item.selected_pts_j)
        chunks.append((first - second).reshape(-1))
    return np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.float64)


def optimize_affines(
    image_ids: list[int],
    observations: list[MatchObservation],
    reference_image: int,
    *,
    sigma_tr: float = 5000.0,
    max_nfev: int = 200,
    verbose: int = 0,
) -> AffineOptimizationResult:
    """Optimize all non-reference affine transforms simultaneously."""

    initial = initialize_affines(image_ids, observations, reference_image, sigma_tr)
    variable_ids = [image_id for image_id in image_ids if image_id != reference_image]
    x0 = np.concatenate([encode_affine_parameters(initial[image_id], sigma_tr) for image_id in variable_ids])
    initial_residual = affine_residuals(x0, image_ids, reference_image, observations, sigma_tr)
    sparsity = _affine_jacobian_sparsity(image_ids, reference_image, observations)
    result = least_squares(
        affine_residuals,
        x0,
        args=(image_ids, reference_image, observations, sigma_tr),
        method="trf",
        jac_sparsity=sparsity,
        x_scale="jac",
        max_nfev=max_nfev,
        verbose=verbose,
    )
    final_residual = affine_residuals(result.x, image_ids, reference_image, observations, sigma_tr)
    return AffineOptimizationResult(
        transforms=_decode_affine_state(result.x, image_ids, reference_image, sigma_tr),
        reference_image=reference_image,
        initial_cost=0.5 * float(initial_residual @ initial_residual),
        final_cost=float(result.cost),
        initial_rmse_px=_point_rmse(initial_residual),
        final_rmse_px=_point_rmse(final_residual),
        nfev=int(result.nfev),
        optimality=float(result.optimality),
        status=int(result.status),
        success=bool(result.success),
        message=str(result.message),
    )


def _decode_affine_state(parameters: np.ndarray, image_ids: list[int], reference_image: int,
                         sigma_tr: float) -> dict[int, np.ndarray]:
    transforms = {reference_image: np.eye(3, dtype=np.float64)}
    offset = 0
    for image_id in image_ids:
        if image_id == reference_image:
            continue
        transforms[image_id] = decode_affine_parameters(parameters[offset:offset + 4], sigma_tr)
        offset += 4
    return transforms


def _warp_affine(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    return values @ transform[:2, :2].T + transform[:2, 2]


def _point_rmse(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(residual.reshape(-1, 2) ** 2, axis=1)))) if residual.size else 0.0


def _affine_jacobian_sparsity(image_ids: list[int], reference_image: int,
                              observations: list[MatchObservation]) -> lil_matrix:
    variable_ids = [image_id for image_id in image_ids if image_id != reference_image]
    columns = {image_id: 4 * index for index, image_id in enumerate(variable_ids)}
    row_count = 2 * sum(item.selected_pts_i.shape[0] for item in observations)
    pattern = lil_matrix((row_count, 4 * len(variable_ids)), dtype=np.int8)
    row = 0
    for item in observations:
        rows = 2 * item.selected_pts_i.shape[0]
        for image_id in (item.image_i, item.image_j):
            if image_id in columns:
                pattern[row:row + rows, columns[image_id]:columns[image_id] + 4] = 1
        row += rows
    return pattern
