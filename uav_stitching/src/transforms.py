"""Homography conventions and parameter encoding shared by optimizers."""

from __future__ import annotations

import numpy as np


def warp_points(homography: np.ndarray, points: np.ndarray, denominator_epsilon: float = 1e-10) -> np.ndarray:
    """Warp image pixels into global mosaic coordinates.

    ``homography`` always maps **image pixel -> global mosaic coordinate**.
    ``points`` is an ``(N,2)`` array of pixel ``(x,y)`` values. The function
    returns Euclidean global ``(X,Y)`` values after homogeneous division and
    raises when a denominator is non-finite or too close to zero.
    """

    matrix = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((values, np.ones(values.shape[0], dtype=np.float64)))
    projected = homogeneous @ matrix.T
    denominator = projected[:, 2]
    if not np.all(np.isfinite(projected)) or np.any(np.abs(denominator) < denominator_epsilon):
        raise ValueError("Homography projection has a non-finite or near-zero denominator")
    return projected[:, :2] / denominator[:, None]


def encode_homography_parameters(homography: np.ndarray, sigma_tr: float) -> np.ndarray:
    """Encode an image-to-global homography as ``[a,b,c',d,e,f',g,h]``.

    Translation variables use ``c'=c/sigma_tr`` and ``f'=f/sigma_tr``. The
    homography is normalized to ``H[2,2]=1`` before encoding.
    """

    if sigma_tr <= 0:
        raise ValueError("sigma_tr must be positive")
    matrix = normalize_homography(homography)
    return np.array(
        [matrix[0, 0], matrix[0, 1], matrix[0, 2] / sigma_tr,
         matrix[1, 0], matrix[1, 1], matrix[1, 2] / sigma_tr,
         matrix[2, 0], matrix[2, 1]],
        dtype=np.float64,
    )


def decode_homography_parameters(parameters: np.ndarray, sigma_tr: float) -> np.ndarray:
    """Decode ``[a,b,c',d,e,f',g,h]`` to image-pixel -> global homography."""

    if sigma_tr <= 0:
        raise ValueError("sigma_tr must be positive")
    values = np.asarray(parameters, dtype=np.float64).reshape(8)
    return np.array(
        [
            [values[0], values[1], values[2] * sigma_tr],
            [values[3], values[4], values[5] * sigma_tr],
            [values[6], values[7], 1.0],
        ],
        dtype=np.float64,
    )


def encode_affine_parameters(homography: np.ndarray, sigma_tr: float) -> np.ndarray:
    """Encode the closest shared-variable similarity as ``[a,b,c',f']``.

    The decoded matrix is ``[[a,b,c],[-b,a,f],[0,0,1]]`` and maps image pixel
    coordinates to global mosaic coordinates.
    """

    matrix = normalize_homography(homography)
    a = 0.5 * (matrix[0, 0] + matrix[1, 1])
    b = 0.5 * (matrix[0, 1] - matrix[1, 0])
    return np.array([a, b, matrix[0, 2] / sigma_tr, matrix[1, 2] / sigma_tr], dtype=np.float64)


def decode_affine_parameters(parameters: np.ndarray, sigma_tr: float) -> np.ndarray:
    """Decode ``[a,b,c',f']`` to the paper's image-to-global affine matrix."""

    values = np.asarray(parameters, dtype=np.float64).reshape(4)
    return np.array(
        [[values[0], values[1], values[2] * sigma_tr],
         [-values[1], values[0], values[3] * sigma_tr],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def normalize_homography(homography: np.ndarray) -> np.ndarray:
    """Return a finite float64 homography normalized to ``H[2,2]=1``."""

    matrix = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(matrix)) or abs(matrix[2, 2]) < 1e-12:
        raise ValueError("Homography is non-finite or cannot be normalized")
    return matrix / matrix[2, 2]


def estimate_similarity(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Least-squares similarity mapping source pixels to target pixels."""

    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 2)
    if source.shape != target.shape or source.shape[0] < 2:
        raise ValueError("At least two paired 2D points are required")
    count = source.shape[0]
    system = np.zeros((2 * count, 4), dtype=np.float64)
    values = np.empty((2 * count,), dtype=np.float64)
    system[0::2, 0] = source[:, 0]
    system[0::2, 1] = -source[:, 1]
    system[0::2, 2] = 1.0
    system[1::2, 0] = source[:, 1]
    system[1::2, 1] = source[:, 0]
    system[1::2, 3] = 1.0
    values[0::2] = target[:, 0]
    values[1::2] = target[:, 1]
    u, v, tx, ty = np.linalg.lstsq(system, values, rcond=None)[0]
    return np.array([[u, -v, tx], [v, u, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def image_corners(width: int, height: int) -> np.ndarray:
    """Return pixel-boundary corners in clockwise order."""

    return np.array([[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]])
