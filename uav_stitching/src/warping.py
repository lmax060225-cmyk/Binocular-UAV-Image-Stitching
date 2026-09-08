"""Transformation sanity checks, canvas geometry, and diagnostic previews."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .transforms import image_corners, warp_points
from .utils import ensure_parent


@dataclass(frozen=True)
class CanvasGeometry:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    width: int
    height: int
    translation: np.ndarray


def transformation_sanity_checks(
    transforms: dict[int, np.ndarray],
    image_shapes: dict[int, tuple[int, int]],
    *,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 20.0,
) -> list[dict[str, object]]:
    """Check finite corners, polygon area, edges, condition, and denominator sign."""

    rows: list[dict[str, object]] = []
    for image_id, transform in sorted(transforms.items()):
        width, height = image_shapes[image_id]
        corners = image_corners(width, height)
        homogeneous = np.column_stack((corners, np.ones(4))) @ transform.T
        denominators = homogeneous[:, 2]
        warnings = []
        if not np.all(np.isfinite(homogeneous)):
            warnings.append("non_finite_projection")
            warped = np.full((4, 2), np.nan)
        elif np.any(np.abs(denominators) < 1e-8) or (np.min(denominators) < 0.0 < np.max(denominators)):
            warnings.append("denominator_near_or_across_zero")
            warped = homogeneous[:, :2] / denominators[:, None]
        else:
            warped = homogeneous[:, :2] / denominators[:, None]
        area = abs(_polygon_area(warped)) if np.all(np.isfinite(warped)) else 0.0
        area_ratio = area / max(1.0, float(width * height))
        edges = np.linalg.norm(np.roll(warped, -1, axis=0) - warped, axis=1) if np.all(np.isfinite(warped)) else np.array([np.inf])
        condition = float(np.linalg.cond(transform))
        if area_ratio < min_area_ratio:
            warnings.append("area_too_small")
        if area_ratio > max_area_ratio:
            warnings.append("area_too_large")
        if not np.all(np.isfinite(edges)) or np.min(edges) < 1.0 or np.max(edges) > 20.0 * max(width, height):
            warnings.append("edge_length_abnormal")
        if not math.isfinite(condition) or condition > 1e10:
            warnings.append("condition_abnormal")
        if max(abs(transform[2, 0]), abs(transform[2, 1])) > 1e-3:
            warnings.append("perspective_extreme")
        rows.append(
            {
                "image_id": image_id,
                "finite": bool(np.all(np.isfinite(warped))),
                "area_ratio": area_ratio,
                "min_edge_px": float(np.min(edges)),
                "max_edge_px": float(np.max(edges)),
                "condition": condition,
                "g": float(transform[2, 0]),
                "h": float(transform[2, 1]),
                "warnings": warnings,
            }
        )
    return rows


def compute_canvas_geometry(
    transforms: dict[int, np.ndarray], image_shapes: dict[int, tuple[int, int]]
) -> CanvasGeometry:
    """Find full-resolution global corner bounds and non-negative canvas translation."""

    all_corners = []
    for image_id, transform in transforms.items():
        width, height = image_shapes[image_id]
        all_corners.append(warp_points(transform, image_corners(width, height)))
    stacked = np.vstack(all_corners)
    min_x, min_y = np.floor(stacked.min(axis=0))
    max_x, max_y = np.ceil(stacked.max(axis=0))
    width = int(max_x - min_x)
    height = int(max_y - min_y)
    translation = np.array([[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]])
    return CanvasGeometry(float(min_x), float(min_y), float(max_x), float(max_y), width, height, translation)


def render_average_preview(
    transforms: dict[int, np.ndarray],
    image_paths: dict[int, Path],
    image_shapes: dict[int, tuple[int, int]],
    output_path: str | Path,
    *,
    scale: float,
    max_pixels: int = 80_000_000,
) -> CanvasGeometry:
    """Render an averaged downsample diagnostic; this is not the final blender."""

    if not 0.0 < scale <= 1.0:
        raise ValueError("Preview scale must be in (0,1]")
    geometry = compute_canvas_geometry(transforms, image_shapes)
    output_width = max(1, int(math.ceil(geometry.width * scale)))
    output_height = max(1, int(math.ceil(geometry.height * scale)))
    if output_width * output_height > max_pixels:
        raise MemoryError(f"Preview canvas {output_width}x{output_height} exceeds {max_pixels} pixels")
    accumulation = np.zeros((output_height, output_width, 3), dtype=np.float32)
    counts = np.zeros((output_height, output_width), dtype=np.float32)
    scaling = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]])
    for image_id in sorted(transforms):
        image = cv2.imread(str(image_paths[image_id]), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not decode image: {image_paths[image_id]}")
        canvas_transform = scaling @ geometry.translation @ transforms[image_id]
        warped = cv2.warpPerspective(image, canvas_transform, (output_width, output_height), flags=cv2.INTER_AREA)
        mask = cv2.warpPerspective(
            np.full(image.shape[:2], 255, dtype=np.uint8), canvas_transform, (output_width, output_height),
            flags=cv2.INTER_NEAREST,
        )
        valid = mask > 0
        accumulation[valid] += warped[valid]
        counts[valid] += 1.0
    preview = np.zeros_like(accumulation, dtype=np.uint8)
    valid = counts > 0
    preview[valid] = np.clip(accumulation[valid] / counts[valid, None], 0, 255).astype(np.uint8)
    destination = Path(output_path)
    ensure_parent(destination)
    if not cv2.imwrite(str(destination), preview, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Could not write preview: {destination}")
    return geometry


def draw_global_footprints(
    transforms: dict[int, np.ndarray],
    image_shapes: dict[int, tuple[int, int]],
    output_path: str | Path,
) -> None:
    """Draw every transformed image quadrilateral with its image ID."""

    geometry = compute_canvas_geometry(transforms, image_shapes)
    width, height = 1800, 1400
    margin = 80
    scale = min((width - 2 * margin) / max(1, geometry.width), (height - 2 * margin) / max(1, geometry.height))
    canvas = np.full((height, width, 3), 255, np.uint8)
    offset_x = (width - geometry.width * scale) / 2.0
    offset_y = (height - geometry.height * scale) / 2.0
    colors = [(0, 114, 178), (213, 94, 0), (0, 158, 115), (204, 121, 167), (230, 159, 0)]
    for image_id, transform in sorted(transforms.items()):
        image_width, image_height = image_shapes[image_id]
        corners = warp_points(transform, image_corners(image_width, image_height))
        pixels = np.column_stack(((corners[:, 0] - geometry.min_x) * scale + offset_x,
                                  (corners[:, 1] - geometry.min_y) * scale + offset_y))
        polygon = np.rint(pixels).astype(np.int32)
        color = colors[image_id % len(colors)]
        cv2.polylines(canvas, [polygon], True, color, 1, cv2.LINE_AA)
        center = tuple(np.rint(pixels.mean(axis=0)).astype(int))
        cv2.putText(canvas, str(image_id), center, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Global transformed footprints ({len(transforms)} images)", (40, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    destination = Path(output_path)
    ensure_parent(destination)
    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"Could not write footprint plot: {destination}")


def _polygon_area(points: np.ndarray) -> float:
    if points.shape != (4, 2):
        return 0.0
    return 0.5 * float(np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1)))
