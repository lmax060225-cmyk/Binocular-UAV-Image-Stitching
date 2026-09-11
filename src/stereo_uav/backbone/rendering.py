"""Rendering for the backbone algorithm."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from . import config
from .geometry import (
    transform_points_projective,
)
from .io import (
    image_pixels,
    imwrite_unicode,
)
from .models import (
    ImageRecord,
)

# ==================== PERSISTENT TWO-BLOCK BACKBONE/CORRECTION OPTIMIZATION END ====================


def compute_canvas(
    images: Sequence[ImageRecord],
    transforms: np.ndarray,
    max_canvas_size: Optional[int] = config.MAX_CANVAS_SIZE,
    verbose: bool = True,
):
    """Compute global canvas bounds from transformed image corners and translate negative

    coordinates into the canvas. An optional preview scale changes rendering only, leaving saved
    transforms in input-pixel coordinates.
    """
    if not images:
        raise ValueError("Cannot compute a canvas without images.")
    if len(images) != len(transforms):
        raise ValueError(
            "The number of images must match the number of transforms."
            f"{len(images)} != {len(transforms)}"
        )
    if not np.all(np.isfinite(transforms)):
        raise RuntimeError("Projective transforms contain NaN or Inf.")

    all_corners = []  # Accumulate the four transformed corners of every image.

    # Associate each image with its image-to-global homography.
    for image, H in zip(images, transforms):
        corners = np.array(
            [
                [0.0, 0.0],  # Top left.
                [image.width - 1.0, 0.0],  # Top right.
                [image.width - 1.0, image.height - 1.0],  # Bottom right.
                [0.0, image.height - 1.0],
            ],  # Bottom left.
            dtype=np.float64,
        )

        # Map corners into the global coordinate system.
        projected_corners = transform_points_projective(H, corners)
        if not np.all(np.isfinite(projected_corners)):
            raise RuntimeError(f"Non-finite projected corners for image: {image.name}")
        all_corners.append(projected_corners)

    # Stack all transformed corners.
    # Stack the transformed corners into an array with four rows per image.
    corners = np.vstack(all_corners)

    # Find minimum global x and y.
    min_xy = np.floor(np.min(corners, axis=0)).astype(np.float64)
    max_xy = np.ceil(np.max(corners, axis=0)).astype(np.float64)

    # Compute the canvas dimensions needed to contain the warped images.
    width = int(max_xy[0] - min_xy[0] + 1)
    height = int(max_xy[1] - min_xy[1] + 1)

    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid canvas size computed from projective transforms.")

    # Translate negative global coordinates into nonnegative canvas coordinates.
    T_canvas = np.array(
        [[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    # Do not scale the preview by default.
    scale = 1.0

    # Use the longest canvas side for the size limit.
    max_side = max(width, height)

    # Downscale oversized previews; None disables the limit.
    if max_canvas_size is not None and max_side > max_canvas_size:
        scale = max_canvas_size / float(max_side)
        if verbose:
            print(
                f"  Canvas {width}x{height} exceeds preview limit="
                f"{max_canvas_size}; preview scale={scale:.4f}"
            )

    preview_width = max(1, int(math.ceil(width * scale)))
    preview_height = max(1, int(math.ceil(height * scale)))

    S = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    return {
        "T_canvas": T_canvas,
        "scale": scale,
        "width": width,
        "height": height,
        "preview_width": preview_width,
        "preview_height": preview_height,
        "preview_transform": S @ T_canvas,  # Translate negative coordinates into the canvas first.
        # Apply the preview scale after canvas translation.
    }


def make_graphcut_seam_finder():
    """Create a Graph-Cut seam finder across supported OpenCV Python interfaces."""
    candidates = []  # Collect available constructor candidates.
    # Each candidate contains an interface name and a constructor.

    if hasattr(cv2, "detail_GraphCutSeamFinder"):
        candidates.append(
            ("cv2.detail_GraphCutSeamFinder", lambda: cv2.detail_GraphCutSeamFinder("COST_COLOR"))
        )

    if hasattr(cv2, "detail") and hasattr(cv2.detail, "GraphCutSeamFinder"):
        candidates.append(
            ("cv2.detail.GraphCutSeamFinder", lambda: cv2.detail.GraphCutSeamFinder("COST_COLOR"))
        )

    if hasattr(cv2, "detail") and hasattr(cv2.detail, "GraphCutSeamFinder_create"):
        candidates.append(
            (
                "cv2.detail.GraphCutSeamFinder_create",
                lambda: cv2.detail.GraphCutSeamFinder_create("COST_COLOR"),
            )
        )

    errors = []

    for name, ctor in candidates:
        try:
            return ctor()
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    raise RuntimeError("Cannot create an OpenCV GraphCutSeamFinder.\n" + "\n".join(errors))


def get_mask_bounding_box(mask: np.ndarray):
    """Return the smallest bounding rectangle of nonzero mask pixels."""

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_0 = int(xs.min())
    x_1 = int(xs.max()) + 1
    y_0 = int(ys.min())
    y_1 = int(ys.max()) + 1
    return x_0, y_0, x_1, y_1


def compose_graphcut_mosaic_preview(
    images: Sequence[ImageRecord],
    transforms: np.ndarray,
    max_canvas_size: Optional[int] = config.MAX_CANVAS_SIZE,
    use_graphcut: Optional[bool] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Compose sequentially: merge the current mosaic and the next warped image using graph-cut

    seam masks when enabled.
    """

    graphcut_enabled = config.USE_GRAPHCUT if use_graphcut is None else bool(use_graphcut)
    blend_mode = "graph-cut" if graphcut_enabled else "hard-overlay"
    if verbose:
        print(f"Composing {blend_mode} mosaic preview")

    # Compute the global rendering canvas.
    # Compute transformed corner bounds, canvas dimensions, translation, and optional preview
    # scaling.

    canvas_info = compute_canvas(
        images,
        transforms,
        max_canvas_size=max_canvas_size,
        verbose=verbose,
    )

    preview_T = canvas_info["preview_transform"]

    size = (
        canvas_info["preview_width"],
        canvas_info["preview_height"],
    )

    # Allocate a black BGR mosaic canvas.
    mosaic = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    # Allocate the corresponding validity mask.
    mosaic_mask = np.zeros((size[1], size[0]), dtype=np.uint8)

    # Create the Graph-Cut seam finder.
    # Use COST_COLOR_GRAD to select a seam based on color and gradient disagreement.
    seam_finder = make_graphcut_seam_finder() if graphcut_enabled else None

    for image, H in tqdm(
        zip(images, transforms),
        total=len(images),
        desc=f"{blend_mode} preview",
        disable=not verbose,
    ):
        H_canvas = preview_T @ H

        # Warp the current image into the global canvas.
        warped = cv2.warpPerspective(
            image_pixels(image),
            H_canvas,
            size,
            flags=cv2.INTER_LINEAR,  # Use bilinear interpolation for image pixels.
        )
        # Pixels outside the warped image are filled with zero.

        # Create an all-valid source mask.
        mask_src = np.full((image.height, image.width), 255, dtype=np.uint8)

        # Warp the validity mask into the canvas.
        warped_mask = cv2.warpPerspective(mask_src, H_canvas, size, flags=cv2.INTER_NEAREST)

        # Binarize the warped mask.
        warped_mask = ((warped_mask > 0).astype(np.uint8)) * 255

        # An empty warped mask means the image contributes no pixels; possible causes include an
        # invalid transform or incorrect canvas bounds.
        if not np.any(warped_mask):
            continue

        # An empty mosaic mask identifies the first contributing image.
        if not np.any(mosaic_mask):
            valid = warped_mask > 0
            mosaic[valid] = warped[valid]
            mosaic_mask[valid] = 255
            continue

        # Without Graph-Cut, later valid pixels overwrite earlier content.
        if not graphcut_enabled:
            valid = warped_mask > 0
            mosaic[valid] = warped[valid]
            mosaic_mask[valid] = 255
            continue

        overlap = (mosaic_mask > 0) & (warped_mask > 0)

        # Paste directly when there is no overlap.
        if not np.any(overlap):
            valid = warped_mask > 0
            mosaic[valid] = warped[valid]
            mosaic_mask[valid] = 255
            continue

        # Only pixels inside the new warped image's support can change.
        # Crop to that support to limit Graph-Cut memory and runtime as the mosaic grows.
        current_mask = warped_mask > 0

        # Find the bounding rectangle of the new valid pixels.
        bbox = get_mask_bounding_box(current_mask)

        if bbox is None:
            continue

        x0, y0, x1, y1 = bbox

        mosaic_roi = mosaic[y0:y1, x0:x1]
        warped_roi = warped[y0:y1, x0:x1]

        mosaic_mask_roi = mosaic_mask[y0:y1, x0:x1]
        warped_mask_roi = warped_mask[y0:y1, x0:x1]

        # GraphCutSeamFinder modifies its input masks in place.
        roi_height, roi_width = mosaic_roi.shape[:2]
        seam_scale = min(1.0, config.MAX_GRAPHCUT_ROI_SIZE / float(max(roi_width, roi_height)))

        if seam_scale < 1.0:
            seam_size = (
                max(1, int(round(roi_width * seam_scale))),
                max(1, int(round(roi_height * seam_scale))),
            )
            imgs_roi = [
                cv2.resize(mosaic_roi, seam_size, interpolation=cv2.INTER_AREA).astype(np.float32),
                cv2.resize(warped_roi, seam_size, interpolation=cv2.INTER_AREA).astype(np.float32),
            ]
            masks_roi = [
                cv2.resize(mosaic_mask_roi, seam_size, interpolation=cv2.INTER_NEAREST),
                cv2.resize(warped_mask_roi, seam_size, interpolation=cv2.INTER_NEAREST),
            ]
        else:
            imgs_roi = [
                mosaic_roi.astype(np.float32),
                warped_roi.astype(np.float32),
            ]
            masks_roi = [
                mosaic_mask_roi.copy(),
                warped_mask_roi.copy(),
            ]

        corners_roi = [
            (0, 0),
            (0, 0),
        ]

        assert seam_finder is not None
        seam_finder.find(
            imgs_roi,
            corners_roi,
            masks_roi,
        )

        # Read the source selection masks after Graph-Cut.
        if seam_scale < 1.0:
            old_keep_roi = (
                cv2.resize(masks_roi[0], (roi_width, roi_height), interpolation=cv2.INTER_NEAREST)
                > 0
            )
            new_keep_roi = (
                cv2.resize(masks_roi[1], (roi_width, roi_height), interpolation=cv2.INTER_NEAREST)
                > 0
            )
        else:
            old_keep_roi = masks_roi[0] > 0
            new_keep_roi = masks_roi[1] > 0

        # Copy the previous mosaic ROI.
        updated_roi = mosaic_roi.copy()

        # Overwrite pixels assigned to the new warped image.
        updated_roi[new_keep_roi] = warped_roi[new_keep_roi]

        # Write the merged ROI back into the canvas.
        mosaic[y0:y1, x0:x1] = updated_roi

        # Update local validity.
        updated_mask_roi = ((old_keep_roi | new_keep_roi).astype(np.uint8)) * 255

        # Write the local mask back into the global mask.
        mosaic_mask[y0:y1, x0:x1] = updated_mask_roi

    return mosaic, canvas_info


def save_graphcut_mosaic_preview(
    images: Sequence[ImageRecord],
    transforms: np.ndarray,
    output_path: Path,
) -> Dict[str, object]:
    """Render and save a mosaic using the existing preview interface."""

    mosaic, canvas_info = compose_graphcut_mosaic_preview(
        images,
        transforms,
        max_canvas_size=config.MAX_CANVAS_SIZE,
        use_graphcut=config.USE_GRAPHCUT,
        verbose=True,
    )
    imwrite_unicode(output_path, mosaic)
    return canvas_info
