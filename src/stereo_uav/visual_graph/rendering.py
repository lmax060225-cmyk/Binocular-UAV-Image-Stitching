"""Rendering for the visual graph algorithm."""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

from .features import (
    read_image,
    write_image,
)
from .geometry import (
    image_order,
    key_text,
    warp_points,
)
from .models import (
    Config,
    ImageKey,
)


def image_corners(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])


def mosaic_bounds(
    nodes: list[ImageKey],
    transforms: dict[ImageKey, np.ndarray],
    shapes: dict[ImageKey, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    corners = []
    for key in nodes:
        local = image_corners(shapes[key])
        H = transforms[key]
        denominators = local @ H[2, :2] + 1.0
        # A pole in an image makes a finite canvas impossible; report, never alter H.
        if denominators.min() <= 0 <= denominators.max() or np.min(np.abs(denominators)) < 1e-10:
            raise ValueError(f"Cannot render {key_text(key)}: projective pole intersects image")
        mapped = warp_points(H, local)
        if not np.isfinite(mapped).all():
            raise ValueError(f"Cannot render non-finite corners: {key_text(key)}")
        corners.append(mapped)
    all_corners = np.vstack(corners)
    return np.floor(all_corners.min(axis=0)), np.ceil(all_corners.max(axis=0))


def canvas_render_scale(bounds, cfg: Config) -> float:
    low, high = bounds
    width, height = np.maximum(high - low + 2, 1.0)
    return min(
        1.0,
        math.sqrt(cfg.MOSAIC_MAX_PIXELS / (width * height)),
        cfg.MOSAIC_MAX_SIDE / width,
        cfg.MOSAIC_MAX_SIDE / height,
    )


def graphcut_merge(
    old: np.ndarray,
    new: np.ndarray,
    old_mask: np.ndarray,
    new_mask: np.ndarray,
    seam_finder,
    max_pixels: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if not np.any(old_mask & new_mask):
        choose_new = new_mask > 0
        old[choose_new] = new[choose_new]
        return old, old_mask | new_mask, False
    ratio = min(1.0, math.sqrt(max_pixels / old_mask.size))
    target = (max(1, round(old.shape[1] * ratio)), max(1, round(old.shape[0] * ratio)))
    resize = target != (old.shape[1], old.shape[0])
    images = [
        cv2.resize(img, target, interpolation=cv2.INTER_AREA) if resize else img
        for img in (old, new)
    ]
    masks = [
        cv2.resize(mask, target, interpolation=cv2.INTER_NEAREST) if resize else mask.copy()
        for mask in (old_mask, new_mask)
    ]
    # UMat handles both bindings that mutate masks and bindings that return masks.
    umat_masks = [cv2.UMat(mask) for mask in masks]
    returned = seam_finder.find(
        [cv2.UMat(img.astype(np.float32)) for img in images], [(0, 0), (0, 0)], umat_masks
    )
    resulting_masks = returned if returned is not None else umat_masks
    seam_new = resulting_masks[1]
    seam_new = seam_new.get() if hasattr(seam_new, "get") else np.asarray(seam_new)
    if resize:
        seam_new = cv2.resize(
            seam_new, (old.shape[1], old.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    choose_new = (new_mask > 0) & ((seam_new > 0) | (old_mask == 0))
    old[choose_new] = new[choose_new]
    return old, old_mask | new_mask, True


def render_mosaic(
    nodes: list[ImageKey],
    paths: dict[ImageKey, Path],
    shapes: dict[ImageKey, tuple[int, int]],
    transforms: dict[ImageKey, np.ndarray],
    output: Path,
    cfg: Config,
    render_scale: float | None = None,
) -> dict:
    started = time.perf_counter()
    bounds = mosaic_bounds(nodes, transforms, shapes)
    low, high = bounds
    scale = canvas_render_scale(bounds, cfg) if render_scale is None else render_scale
    width, height = np.ceil((high - low) * scale + 2).astype(int)
    if width >= 32767 or height >= 32767:
        raise ValueError("OpenCV canvas size exceeds supported limit")
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas_mask = np.zeros((height, width), np.uint8)
    view = np.array([[scale, 0.0, -low[0] * scale], [0.0, scale, -low[1] * scale], [0.0, 0.0, 1.0]])
    seam_finder = None
    if hasattr(cv2, "detail_GraphCutSeamFinder"):
        try:
            seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
        except cv2.error as error:
            print(f"WARNING: GraphCut unavailable: {error}; fallback hard overlay", flush=True)
    if seam_finder is None:
        print("WARNING: GraphCut unavailable; fallback hard overlay", flush=True)
    seam_count = fallback_count = 0
    for index, key in enumerate(sorted(nodes, key=image_order)):
        image = read_image(paths[key])
        render_H = view @ transforms[key]
        corners = warp_points(render_H, image_corners(shapes[key]))
        # A margin includes the existing mosaic just outside the new image for seam terminals.
        x0, y0 = np.maximum(np.floor(corners.min(axis=0)).astype(int) - 8, 0)
        x1, y1 = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + 9, [width, height])
        shift = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
        local_H = shift @ render_H
        size = (int(x1 - x0), int(y1 - y0))
        warped = cv2.warpPerspective(image, local_H, size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(
            np.full(image.shape[:2], 255, np.uint8), local_H, size, flags=cv2.INTER_NEAREST
        )
        old, old_mask = canvas[y0:y1, x0:x1], canvas_mask[y0:y1, x0:x1]
        if seam_finder is not None:
            try:
                merged, union, used = graphcut_merge(
                    old, warped, old_mask, mask, seam_finder, cfg.GRAPHCUT_MAX_PIXELS
                )
                seam_count += int(used)
            except cv2.error as error:
                print(
                    f"WARNING: GraphCut failed for {key_text(key)}: {error}; "
                    "fallback hard overlay for this image",
                    flush=True,
                )
                fallback_count += 1
                merged, union = old, old_mask | mask
                merged[mask > 0] = warped[mask > 0]
        else:
            fallback_count += 1
            merged, union = old, old_mask | mask
            merged[mask > 0] = warped[mask > 0]
        canvas[y0:y1, x0:x1], canvas_mask[y0:y1, x0:x1] = merged, union
        if (index + 1) % 10 == 0 or index + 1 == len(nodes):
            print(
                f"Mosaic {output.name}: {index + 1}/{len(nodes)}, "
                f"GraphCut seams={seam_count}, elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    write_image(output, canvas)
    write_image(output.with_name(output.stem + "_mask.png"), canvas_mask)
    preview_scale = min(1.0, 1800 / max(width, height))
    preview = cv2.resize(
        canvas,
        (round(width * preview_scale), round(height * preview_scale)),
        interpolation=cv2.INTER_AREA,
    )
    write_image(output.with_name(output.stem + "_preview.jpg"), preview)
    return dict(
        file=str(output),
        width=int(width),
        height=int(height),
        render_scale=scale,
        input_coordinate_units="original image pixels",
        graphcut_available=seam_finder is not None,
        graphcut_seams=seam_count,
        hard_overlay_fallbacks=fallback_count,
        graphcut_max_pixels=cfg.GRAPHCUT_MAX_PIXELS,
        valid_pixels=int(np.count_nonzero(canvas_mask)),
        canvas_coverage=float(np.count_nonzero(canvas_mask) / canvas_mask.size),
        view_transform=view.tolist(),
        runtime=time.perf_counter() - started,
    )
