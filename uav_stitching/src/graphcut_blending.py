"""Sequential frame-to-frame Graph-Cut blending on a bounded global canvas.

Registration follows the paper's global model.  The seam solver is the OpenCV
``GraphCutSeamFinder`` engineering equivalent because the cited paper does not
publish all implementation details of its seam energy.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .optimization_data import MatchObservation
from .utils import ensure_parent
from .warping import CanvasGeometry, compute_canvas_geometry


@dataclass(frozen=True)
class BlendStep:
    step: int
    image_id: int
    overlap_pixels_at_seam_scale: int
    seam_method: str
    elapsed_seconds: float


@dataclass(frozen=True)
class BlendResult:
    output_path: Path
    mask_path: Path
    order: tuple[int, ...]
    full_resolution_geometry: CanvasGeometry
    output_width: int
    output_height: int
    output_scale: float
    seam_scale: float
    covered_pixels: int
    coverage_ratio: float
    steps: tuple[BlendStep, ...]


def strongest_overlap_order(
    image_ids: list[int], observations: list[MatchObservation], reference_image: int
) -> list[int]:
    """Return a deterministic maximum-overlap tree traversal from the reference.

    Each next image is attached by the currently strongest valid RANSAC edge
    crossing from the accumulated mosaic to an unseen image.  This is an
    implementation choice for the paper's unspecified frame insertion order.
    """

    requested = set(image_ids)
    if reference_image not in requested:
        raise ValueError("Reference image is not in image_ids")
    order = [reference_image]
    selected = {reference_image}
    while selected != requested:
        candidates: list[tuple[float, float, int, int, int]] = []
        for item in observations:
            if item.image_i not in requested or item.image_j not in requested:
                continue
            if (item.image_i in selected) ^ (item.image_j in selected):
                new_id = item.image_j if item.image_i in selected else item.image_i
                old_id = item.image_i if item.image_i in selected else item.image_j
                candidates.append((-float(item.inlier_count), item.pairwise_rmse_px, new_id, old_id, new_id))
        if not candidates:
            missing = sorted(requested - selected)
            raise ValueError(f"Valid match graph cannot reach images: {missing}")
        new_id = min(candidates)[-1]
        selected.add(new_id)
        order.append(new_id)
    return order


def graphcut_available() -> bool:
    """Report whether the active OpenCV build exposes GraphCutSeamFinder."""

    return hasattr(cv2, "detail_GraphCutSeamFinder")


def sequential_graphcut_blend(
    transforms: dict[int, np.ndarray],
    image_paths: dict[int, Path],
    image_shapes: dict[int, tuple[int, int]],
    order: list[int],
    output_path: str | Path,
    mask_path: str | Path,
    seam_debug_dir: str | Path,
    *,
    output_scale: float = 0.25,
    seam_scale: float = 0.10,
    max_output_pixels: int = 50_000_000,
    seam_cost: str = "COST_COLOR_GRAD",
) -> BlendResult:
    """Warp and blend images in ``A+B -> AB; AB+C -> ABC`` order.

    Homographies map source-image pixels to the full-resolution global mosaic.
    Seams are estimated on a smaller uint8/float32 canvas to bound Graph-Cut
    memory, then transferred by nearest-neighbor scaling to the uint8 output
    canvas.  There is no full-resolution float RGB accumulator.
    """

    if not graphcut_available():
        raise RuntimeError("This OpenCV build lacks cv2.detail_GraphCutSeamFinder")
    if not order or set(order) != set(transforms) or len(order) != len(transforms):
        raise ValueError("Order must contain every transformed image exactly once")
    if not 0.0 < seam_scale <= output_scale <= 1.0:
        raise ValueError("Require 0 < seam_scale <= output_scale <= 1")

    geometry = compute_canvas_geometry(transforms, image_shapes)
    output_width = max(1, int(math.ceil(geometry.width * output_scale)))
    output_height = max(1, int(math.ceil(geometry.height * output_scale)))
    seam_width = max(1, int(math.ceil(geometry.width * seam_scale)))
    seam_height = max(1, int(math.ceil(geometry.height * seam_scale)))
    if output_width * output_height > max_output_pixels:
        raise MemoryError(
            f"Output canvas {output_width}x{output_height}={output_width * output_height:,} pixels "
            f"exceeds configured limit {max_output_pixels:,}"
        )

    output = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    output_mask = np.zeros((output_height, output_width), dtype=np.uint8)
    seam_mosaic = np.zeros((seam_height, seam_width, 3), dtype=np.uint8)
    seam_mosaic_mask = np.zeros((seam_height, seam_width), dtype=np.uint8)
    output_scaling = np.diag([output_scale, output_scale, 1.0])
    seam_scaling = np.diag([seam_scale, seam_scale, 1.0])
    debug_directory = Path(seam_debug_dir)
    debug_directory.mkdir(parents=True, exist_ok=True)
    steps: list[BlendStep] = []

    for step, image_id in enumerate(order):
        started = time.perf_counter()
        image = cv2.imread(str(image_paths[image_id]), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not decode image: {image_paths[image_id]}")
        source_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        output_transform = output_scaling @ geometry.translation @ transforms[image_id]
        seam_transform = seam_scaling @ geometry.translation @ transforms[image_id]
        warped_output = cv2.warpPerspective(
            image, output_transform, (output_width, output_height), flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_output_mask = cv2.warpPerspective(
            source_mask, output_transform, (output_width, output_height), flags=cv2.INTER_NEAREST,
        )
        warped_seam = cv2.warpPerspective(
            image, seam_transform, (seam_width, seam_height), flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_seam_mask = cv2.warpPerspective(
            source_mask, seam_transform, (seam_width, seam_height), flags=cv2.INTER_NEAREST,
        )

        if step == 0:
            new_take_seam = warped_seam_mask > 0
            seam_method = "initial"
            overlap_pixels = 0
        else:
            overlap = (seam_mosaic_mask > 0) & (warped_seam_mask > 0)
            overlap_pixels = int(np.count_nonzero(overlap))
            if overlap_pixels:
                new_take_seam, seam_method = _find_sequential_seam(
                    seam_mosaic, seam_mosaic_mask, warped_seam, warped_seam_mask, seam_cost
                )
            else:
                # The order is graph-connected, but rounding or a marginal
                # registration can eliminate raster overlap at seam scale.
                new_take_seam = warped_seam_mask > 0
                seam_method = "no_raster_overlap"

        new_take_output = cv2.resize(
            (new_take_seam.astype(np.uint8) * 255), (output_width, output_height),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        output_valid = warped_output_mask > 0
        # Always preserve new-only support; use the transferred cut in overlap.
        take_output = output_valid & ((output_mask == 0) | new_take_output)
        output[take_output] = warped_output[take_output]
        output_mask[output_valid] = 255

        seam_valid = warped_seam_mask > 0
        take_seam = seam_valid & ((seam_mosaic_mask == 0) | new_take_seam)
        seam_mosaic[take_seam] = warped_seam[take_seam]
        seam_mosaic_mask[seam_valid] = 255

        _write_seam_debug(
            debug_directory / f"step_{step:03d}_image_{image_id:04d}.jpg",
            seam_mosaic,
            seam_mosaic_mask,
            new_take_seam,
        )
        steps.append(
            BlendStep(
                step=step,
                image_id=image_id,
                overlap_pixels_at_seam_scale=overlap_pixels,
                seam_method=seam_method,
                elapsed_seconds=time.perf_counter() - started,
            )
        )

    destination = Path(output_path)
    mask_destination = Path(mask_path)
    _write_image_atomic(destination, output, [cv2.IMWRITE_JPEG_QUALITY, 95])
    _write_image_atomic(mask_destination, output_mask, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    covered = int(np.count_nonzero(output_mask))
    return BlendResult(
        output_path=destination,
        mask_path=mask_destination,
        order=tuple(order),
        full_resolution_geometry=geometry,
        output_width=output_width,
        output_height=output_height,
        output_scale=output_scale,
        seam_scale=seam_scale,
        covered_pixels=covered,
        coverage_ratio=covered / float(output_width * output_height),
        steps=tuple(steps),
    )


def _find_sequential_seam(
    mosaic: np.ndarray,
    mosaic_mask: np.ndarray,
    new_image: np.ndarray,
    new_mask: np.ndarray,
    seam_cost: str,
) -> tuple[np.ndarray, str]:
    """Run OpenCV Graph-Cut on the new image bbox, with a safe fallback."""

    locations = cv2.findNonZero(new_mask)
    if locations is None:
        return np.zeros_like(new_mask, dtype=bool), "empty_new_mask"
    x, y, width, height = cv2.boundingRect(locations)
    current_crop = mosaic[y:y + height, x:x + width]
    new_crop = new_image[y:y + height, x:x + width]
    current_mask_crop = mosaic_mask[y:y + height, x:x + width]
    new_mask_crop = new_mask[y:y + height, x:x + width]
    try:
        finder = cv2.detail_GraphCutSeamFinder(seam_cost)
        returned = finder.find(
            [cv2.UMat(current_crop.astype(np.float32)), cv2.UMat(new_crop.astype(np.float32))],
            [(0, 0), (0, 0)],
            [cv2.UMat(current_mask_crop.copy()), cv2.UMat(new_mask_crop.copy())],
        )
        cut_current = returned[0].get() > 0
        cut_new = returned[1].get() > 0
        new_take_crop = cut_new & ~cut_current
        unresolved = cut_new & cut_current
        if np.any(unresolved):
            current_distance = cv2.distanceTransform((current_mask_crop > 0).astype(np.uint8), cv2.DIST_L2, 3)
            new_distance = cv2.distanceTransform((new_mask_crop > 0).astype(np.uint8), cv2.DIST_L2, 3)
            new_take_crop |= unresolved & (new_distance > current_distance)
        # Non-overlap belongs to the new image regardless of Graph-Cut output.
        new_take_crop |= (new_mask_crop > 0) & (current_mask_crop == 0)
        method = "opencv_graphcut"
    except (cv2.error, RuntimeError, ValueError):
        current_distance = cv2.distanceTransform((current_mask_crop > 0).astype(np.uint8), cv2.DIST_L2, 3)
        new_distance = cv2.distanceTransform((new_mask_crop > 0).astype(np.uint8), cv2.DIST_L2, 3)
        new_take_crop = (new_mask_crop > 0) & (
            (current_mask_crop == 0) | (new_distance > current_distance)
        )
        method = "distance_fallback"
    new_take = np.zeros_like(new_mask, dtype=bool)
    new_take[y:y + height, x:x + width] = new_take_crop
    return new_take, method


def _write_seam_debug(path: Path, mosaic: np.ndarray, mask: np.ndarray, new_take: np.ndarray) -> None:
    preview = mosaic.copy()
    boundary = cv2.morphologyEx(new_take.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    preview[boundary] = (0, 0, 255)
    preview[mask == 0] = 0
    _write_image_atomic(path, preview, [cv2.IMWRITE_JPEG_QUALITY, 88])


def _write_image_atomic(path: Path, image: np.ndarray, parameters: list[int]) -> None:
    ensure_parent(path)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        if not cv2.imwrite(str(temporary), image, parameters):
            raise OSError(f"Could not write image: {temporary}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
