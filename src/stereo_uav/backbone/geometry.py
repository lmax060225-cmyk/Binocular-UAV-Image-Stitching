"""Geometry for the backbone algorithm."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config
from .models import (
    ImageRecord,
    PairMatch_Edge,
)


def estimate_local_similarity(edge: Optional[PairMatch_Edge]) -> Optional[np.ndarray]:
    """Estimate a local similarity from the spatially selected points of an edge."""

    if edge is None or len(edge.selected_pts_i) < 2:
        return None

    # Use cv2.estimateAffinePartial2D on the selected correspondences.

    A, _ = cv2.estimateAffinePartial2D(
        edge.selected_pts_i.astype(np.float64),
        edge.selected_pts_j.astype(np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=config.RANSAC_REPROJ_THRESH,
    )

    if A is None:
        return None

    # Start with a homogeneous 3x3 identity matrix.
    H = np.eye(3, dtype=np.float64)
    # Insert the estimated affine rows.
    H[:2, :] = A
    return H


# ==================== RIGID STEREO INITIALIZATION ====================
def project_to_similarity(H: np.ndarray) -> np.ndarray:
    """Project a 3x3 transform to the implementation's 2D similarity convention."""

    H = np.asarray(H, dtype=np.float64)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    a = 0.5 * (H[0, 0] + H[1, 1])
    b = 0.5 * (H[0, 1] - H[1, 0])
    c = H[0, 2]
    f = H[1, 2]

    return np.array(
        [
            [a, b, c],
            [-b, a, f],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def similarity_components(
    H: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """Extract positive scale, global image-x-axis angle, and unit direction [cos(theta),

    sin(theta)].
    """

    a = float(H[0, 0])
    b = float(H[0, 1])

    scale = max(math.hypot(a, b), config.PROJECTIVE_REG_EPS)

    # Matrix convention:
    #   [[a,  b],
    #    [-b, a]]
    # The first column [a, -b]^T is the transformed image-x direction.
    direction = np.array(
        [a / scale, -b / scale],
        dtype=np.float64,
    )
    angle = float(math.atan2(direction[1], direction[0]))
    return scale, angle, direction


def normalize_direction(direction: np.ndarray) -> np.ndarray:
    """Normalize a 2D direction, falling back to the global x axis when degenerate."""

    direction = np.asarray(direction, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return direction / norm


def average_directions(
    direction_i: np.ndarray,
    direction_j: np.ndarray,
) -> np.ndarray:
    """Average two unit directions by normalizing their sum. Fall back to direction_i when the sum

    degenerates; stereo directions are expected to be close.
    """

    direction_i = normalize_direction(direction_i)
    direction_j = normalize_direction(direction_j)

    merged = direction_i + direction_j
    if float(np.linalg.norm(merged)) < 1e-12:
        return direction_i.copy()
    return normalize_direction(merged)


def similarity_linear_from_scale_direction(
    scale: float,
    direction: np.ndarray,
) -> np.ndarray:
    """Construct a 2x2 similarity linear part from shared scale and image-x-axis direction."""

    direction = normalize_direction(direction)
    ux, uy = float(direction[0]), float(direction[1])

    # Standard rotation convention:
    #   s [[ cos(theta), -sin(theta)],
    #      [ sin(theta),  cos(theta)]]
    return float(scale) * np.array(
        [
            [ux, -uy],
            [uy, ux],
        ],
        dtype=np.float64,
    )


def image_center_point(image: ImageRecord) -> np.ndarray:
    """Return the image center in pixel coordinates."""

    return np.array(
        [
            0.5 * (image.width - 1.0),
            0.5 * (image.height - 1.0),
        ],
        dtype=np.float64,
    )


def replace_similarity_linear_preserve_center(
    H: np.ndarray,
    image: ImageRecord,
    new_linear: np.ndarray,
) -> np.ndarray:
    """Replace the similarity linear part while preserving the mapped image center to avoid

    translation jumps during stereo commonization.
    """

    H = project_to_similarity(H)
    center = image_center_point(image)

    old_global_center = H[:2, :2] @ center + H[:2, 2]

    new_H = H.copy()
    new_H[:2, :2] = np.asarray(new_linear, dtype=np.float64)
    new_H[:2, 2] = old_global_center - new_H[:2, :2] @ center
    return new_H


# ==================== RIGID STEREO INITIALIZATION END ====================


def pack_affine_params(transforms: np.ndarray) -> np.ndarray:
    """Pack similarity parameters for images 1 through N-1 into a least-squares vector."""

    params = []

    # Scale translations c and f by sigma before optimization.
    for H in transforms[1:]:
        params.extend(
            [
                H[0, 0],
                H[0, 1],
                H[0, 2] / config.TRANSLATION_SCALE,
                H[1, 2] / config.TRANSLATION_SCALE,
            ]
        )
    return np.array(params, dtype=np.float64)


def unpack_affine_params(params: np.ndarray, num_images: int) -> np.ndarray:
    """Recover global similarities, fixing image 0 to identity."""

    transforms = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], num_images, axis=0)
    for idx in range(1, num_images):
        k = (idx - 1) * 4
        a, b, c_scaled, f_scaled = params[k : k + 4]
        # Similarity constraint from the paper: a=e and b=-d. This preserves
        # rotation + uniform scale + translation before projective refinement.
        transforms[idx] = np.array(
            [
                [a, b, c_scaled * config.TRANSLATION_SCALE],
                [-b, a, f_scaled * config.TRANSLATION_SCALE],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    return transforms


def transform_points_affine(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Append a homogeneous unit coordinate to the N-by-2 point array."""
    hom = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    return (H @ hom.T).T[:, :2]


def wrap_angle(angle: float) -> float:
    """Wrap an angle difference to [-pi, pi]."""
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def homography_jacobian(H: np.ndarray, x: float, y: float) -> np.ndarray:
    """Evaluate the 2x2 Jacobian of a homography at the specified pixel position."""
    a, b, c = H[0]
    d, e, f = H[1]
    g, h, _ = H[2]

    denominator = g * x + h * y + 1.0
    if abs(denominator) < 1e-9:
        denominator = -1e-9 if denominator < 0.0 else 1e-9

    numerator_u = a * x + b * y + c
    numerator_v = d * x + e * y + f
    denominator2 = denominator * denominator

    return np.array(
        [
            [
                (a * denominator - g * numerator_u) / denominator2,
                (b * denominator - h * numerator_u) / denominator2,
            ],
            [
                (d * denominator - g * numerator_v) / denominator2,
                (e * denominator - h * numerator_v) / denominator2,
            ],
        ],
        dtype=np.float64,
    )


def local_rotation_angle(H: np.ndarray, image: ImageRecord) -> float:
    """Extract the closest-rotation angle from the Jacobian at the image center."""
    J = homography_jacobian(
        H,
        0.5 * (image.width - 1.0),
        0.5 * (image.height - 1.0),
    )
    return float(math.atan2(J[1, 0] - J[0, 1], J[0, 0] + J[1, 1]))


def similarity_rotation_angle(H: np.ndarray) -> float:
    """Return the position-independent rotation angle of a similarity transform."""
    return float(math.atan2(H[1, 0] - H[0, 1], H[0, 0] + H[1, 1]))


def normalize_projective_homography(H: np.ndarray) -> np.ndarray:
    """Validate and normalize a homography to H[2, 2] == 1."""

    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Homography must have shape (3, 3), got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Homography contains NaN or Inf.")

    scale = float(matrix[2, 2])
    if abs(scale) < config.PROJECTIVE_REG_EPS:
        raise ValueError("Homography cannot be normalized because H[2, 2] is too small.")

    normalized = matrix.copy() / scale
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Normalized Homography contains NaN or Inf.")
    return normalized


def _pack_one_projective_homography(H: np.ndarray) -> List[float]:
    """Pack an H33=1 homography into eight optimization parameters."""

    H = normalize_projective_homography(H)
    return [
        float(H[0, 0]),
        float(H[0, 1]),
        float(H[0, 2]) / config.TRANSLATION_SCALE,
        float(H[1, 0]),
        float(H[1, 1]),
        float(H[1, 2]) / config.TRANSLATION_SCALE,
        float(H[2, 0]),
        float(H[2, 1]),
    ]


def _unpack_one_projective_homography(values: Sequence[float]) -> np.ndarray:
    """Unpack eight optimization parameters into an H33=1 homography."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size != 8:
        raise ValueError(f"Expected 8 projective parameters, got {values.size}.")

    a, b, c_scaled, d, e, f_scaled, g, h = values
    return np.array(
        [
            [a, b, c_scaled * config.TRANSLATION_SCALE],
            [d, e, f_scaled * config.TRANSLATION_SCALE],
            [g, h, 1.0],
        ],
        dtype=np.float64,
    )


def pack_variable_projective_params(
    transforms: np.ndarray,
    variable_indices: Sequence[int],
) -> np.ndarray:
    """Pack only the homographies selected by variable_indices."""

    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (3, 3):
        raise ValueError("transforms must have shape (N, 3, 3).")

    indices = [int(idx) for idx in variable_indices]
    if len(indices) != len(set(indices)):
        raise ValueError("variable_indices contains duplicate indices.")

    params: List[float] = []
    for idx in indices:
        if idx < 0 or idx >= len(transforms):
            raise IndexError(f"Variable index {idx} is outside the transform array.")
        params.extend(_pack_one_projective_homography(transforms[idx]))
    return np.asarray(params, dtype=np.float64)


def unpack_variable_projective_params(
    params: np.ndarray,
    base_transforms: np.ndarray,
    variable_indices: Sequence[int],
) -> np.ndarray:
    """Update variable homographies in base_transforms; keep all other transforms fixed."""

    base = np.asarray(base_transforms, dtype=np.float64)
    if base.ndim != 3 or base.shape[1:] != (3, 3):
        raise ValueError("base_transforms must have shape (N, 3, 3).")

    indices = [int(idx) for idx in variable_indices]
    if len(indices) != len(set(indices)):
        raise ValueError("variable_indices contains duplicate indices.")

    values = np.asarray(params, dtype=np.float64).reshape(-1)
    expected = 8 * len(indices)
    if values.size != expected:
        raise ValueError(f"Expected {expected} projective variables, got {values.size}.")

    transforms = base.copy()
    for slot, idx in enumerate(indices):
        if idx < 0 or idx >= len(transforms):
            raise IndexError(f"Variable index {idx} is outside the transform array.")
        k = 8 * slot
        transforms[idx] = _unpack_one_projective_homography(values[k : k + 8])
    return transforms


def compose_similarity_corrections(
    backbones: np.ndarray,
    corrections: np.ndarray,
) -> np.ndarray:
    """Compose G_i = S_i @ C_i and normalize H33 for every node."""

    backbones = np.asarray(backbones, dtype=np.float64)
    corrections = np.asarray(corrections, dtype=np.float64)
    if backbones.shape != corrections.shape or backbones.ndim != 3 or backbones.shape[1:] != (3, 3):
        raise ValueError("backbones and corrections must both have shape (N, 3, 3).")

    return np.stack(
        [normalize_projective_homography(S @ C) for S, C in zip(backbones, corrections)],
        axis=0,
    )


def project_homography_to_similarity_backbone(
    H: np.ndarray,
    image: ImageRecord,
) -> np.ndarray:
    """Project a homography to its closest center-Jacobian similarity while preserving the mapped

    center. The result has linear part [[a, b], [-b, a]] and bottom row [0, 0, 1], so projective
    terms do not enter the propagation chain.
    """

    H = normalize_projective_homography(H)
    center = image_center_point(image)
    mapped_center = transform_points_projective(
        H,
        center.reshape(1, 2),
    )[0]

    J = homography_jacobian_general(
        H,
        float(center[0]),
        float(center[1]),
    )
    if J is None:
        raise ValueError("Cannot project a locally singular Homography to a similarity backbone.")

    # Frobenius-nearest similarity with linear part [[a, b], [-b, a]].
    a = 0.5 * float(J[0, 0] + J[1, 1])
    b = 0.5 * float(J[0, 1] - J[1, 0])
    scale = math.hypot(a, b)
    if not math.isfinite(scale) or scale <= config.PROJECTIVE_REG_EPS:
        raise ValueError("Homography center Jacobian has a degenerate similarity component.")

    linear = np.array(
        [
            [a, b],
            [-b, a],
        ],
        dtype=np.float64,
    )
    translation = mapped_center - linear @ center
    backbone = np.array(
        [
            [a, b, float(translation[0])],
            [-b, a, float(translation[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return normalize_projective_homography(backbone)


def decompose_global_transforms(
    transforms: np.ndarray,
    images: Sequence[ImageRecord],
    backbone_template: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose G_i = S_i @ C_i. Without a template, extract S_i from the center Jacobian; with a

    template, preserve S_i and compute C_i = inv(S_i) @ G_i.
    """

    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (3, 3):
        raise ValueError("transforms must have shape (N, 3, 3).")
    if len(transforms) != len(images):
        raise ValueError("images and transforms must have equal length.")

    if backbone_template is None:
        backbones = np.stack(
            [
                project_homography_to_similarity_backbone(H, image)
                for H, image in zip(transforms, images)
            ],
            axis=0,
        )
    else:
        backbones = np.asarray(backbone_template, dtype=np.float64).copy()
        if backbones.shape != transforms.shape:
            raise ValueError("backbone_template must match transforms shape.")
        backbones = np.stack(
            [normalize_projective_homography(S) for S in backbones],
            axis=0,
        )

    corrections: List[np.ndarray] = []
    for S, G in zip(backbones, transforms):
        inverse_backbone = np.linalg.inv(S)
        correction = normalize_projective_homography(
            inverse_backbone @ normalize_projective_homography(G)
        )
        corrections.append(correction)

    corrections_array = np.stack(corrections, axis=0)
    # Verify that recomposition numerically recovers the original G.
    reconstructed = compose_similarity_corrections(backbones, corrections_array)
    normalized_input = np.stack(
        [normalize_projective_homography(H) for H in transforms],
        axis=0,
    )
    if not np.allclose(reconstructed, normalized_input, rtol=1e-8, atol=1e-8):
        raise RuntimeError("Similarity-backbone decomposition failed to reconstruct G.")

    return backbones, corrections_array


def rebase_variable_global_states(
    global_transforms: np.ndarray,
    old_backbones: np.ndarray,
    images: Sequence[ImageRecord],
    variable_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-extract similarity backbones only for variable nodes without changing G_i. Transfer

    similarity components into S_i so the next block propagates S_i while keeping projective
    corrections local.
    """

    global_transforms = np.asarray(global_transforms, dtype=np.float64)
    backbones = np.asarray(old_backbones, dtype=np.float64).copy()
    if global_transforms.shape != backbones.shape:
        raise ValueError("global_transforms and old_backbones must have equal shape.")
    if len(images) != len(global_transforms):
        raise ValueError("images and transforms must have equal length.")

    variable_set = {int(idx) for idx in variable_indices}
    for idx in variable_set:
        if idx < 0 or idx >= len(global_transforms):
            raise IndexError(f"Variable index {idx} is outside global_transforms.")
        backbones[idx] = project_homography_to_similarity_backbone(
            global_transforms[idx],
            images[idx],
        )

    backbones, corrections = decompose_global_transforms(
        global_transforms,
        images,
        backbone_template=backbones,
    )
    reconstructed = compose_similarity_corrections(backbones, corrections)
    return reconstructed, backbones, corrections


def transform_points_projective(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Transform 2D points projectively: q = H [x, y, 1]^T and p_prime = q[:2] / q[2]."""

    hom = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    warped = (H @ hom.T).T
    w = warped[:, 2]
    safe_w = np.where(np.abs(w) < 1e-9, np.where(w < 0.0, -1e-9, 1e-9), w)
    return warped[:, :2] / safe_w[:, None]


def projective_local_similarity_components(
    H: np.ndarray,
    image: ImageRecord,
) -> Tuple[float, float, np.ndarray]:
    """Extract local similarity information from the center Jacobian: RMS singular-value scale,

    closest-rotation angle, and local unit x direction.
    """

    center = image_center_point(image)
    J = homography_jacobian(
        H,
        float(center[0]),
        float(center[1]),
    )

    u_norm2 = float(J[0, 0] * J[0, 0] + J[1, 0] * J[1, 0])
    v_norm2 = float(J[0, 1] * J[0, 1] + J[1, 1] * J[1, 1])

    scale = math.sqrt(
        max(
            0.5 * (u_norm2 + v_norm2),
            config.PROJECTIVE_REG_EPS,
        )
    )

    angle = float(
        math.atan2(
            J[1, 0] - J[0, 1],
            J[0, 0] + J[1, 1],
        )
    )

    direction = np.array(
        [
            math.cos(angle),
            math.sin(angle),
        ],
        dtype=np.float64,
    )

    return scale, angle, direction


def projective_stereo_pair_frame(
    transforms: Sequence[np.ndarray],
    images: Sequence[ImageRecord],
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
    """Return the projective stereo frame evaluated at the image centers."""

    scale_i, angle_i, direction_i = projective_local_similarity_components(
        transforms[i],
        images[i],
    )
    scale_j, angle_j, direction_j = projective_local_similarity_components(
        transforms[j],
        images[j],
    )

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


def make_image_normalization_matrix(width: float, height: float) -> np.ndarray:
    """Map pixel coordinates to the fixed normalized domain [-1, 1]^2."""

    width = float(width)
    height = float(height)
    if not math.isfinite(width) or not math.isfinite(height) or width <= 1.0 or height <= 1.0:
        raise ValueError(
            "Image dimensions must be finite and greater than one, got "
            f"width={width}, height={height}"
        )

    return np.array(
        [
            [2.0 / (width - 1.0), 0.0, -1.0],
            [0.0, 2.0 / (height - 1.0), -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def normalize_homography_for_safety(
    H: np.ndarray,
    source_image: ImageRecord,
    reference_image: ImageRecord,
) -> np.ndarray:
    """Express a source-to-reference homography in resolution-independent normalized coordinates."""

    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"H must have shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Homography contains NaN or Inf.")

    N_src = make_image_normalization_matrix(
        source_image.width,
        source_image.height,
    )
    N_ref = make_image_normalization_matrix(
        reference_image.width,
        reference_image.height,
    )
    H_bar = N_ref @ matrix @ np.linalg.inv(N_src)

    # Homogeneous scale is arbitrary; normalize only when h33 is safe.
    h33 = float(H_bar[2, 2])
    if math.isfinite(h33) and abs(h33) > config.PROJECTIVE_REG_EPS:
        H_bar = H_bar / h33
    return np.asarray(H_bar, dtype=np.float64)


def homography_jacobian_general(
    H: np.ndarray,
    x: float,
    y: float,
) -> Optional[np.ndarray]:
    """Evaluate the exact 2x2 Jacobian of a general 3x3 homography; return None if degenerate."""

    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return None

    h11, h12, h13 = matrix[0]
    h21, h22, h23 = matrix[1]
    h31, h32, h33 = matrix[2]
    x = float(x)
    y = float(y)

    w = h31 * x + h32 * y + h33
    if not math.isfinite(w) or abs(w) <= config.PROJECTIVE_REG_EPS:
        return None

    nu = h11 * x + h12 * y + h13
    nv = h21 * x + h22 * y + h23
    if not math.isfinite(nu) or not math.isfinite(nv):
        return None

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        inv_w2 = 1.0 / (w * w)
        J = np.array(
            [
                [
                    (h11 * w - h31 * nu) * inv_w2,
                    (h12 * w - h32 * nu) * inv_w2,
                ],
                [
                    (h21 * w - h31 * nv) * inv_w2,
                    (h22 * w - h32 * nv) * inv_w2,
                ],
            ],
            dtype=np.float64,
        )

    return J if np.all(np.isfinite(J)) else None
