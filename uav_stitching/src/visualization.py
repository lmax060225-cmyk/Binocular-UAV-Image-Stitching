"""Static OpenCV diagnostic visualization for the GPS registration graph."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .gps import ImagePosition
from .neighbor_graph import Pair
from .utils import ensure_parent


def plot_gps_neighbor_graph(
    positions: list[ImagePosition],
    pairs: list[Pair],
    output_path: str | Path,
    *,
    method: str,
) -> None:
    """Plot ENU locations, capture labels, and candidate edges to a PNG.

    OpenCV drawing is used instead of Matplotlib because the supplied Conda
    environment has a fatal BLAS DLL conflict inside Matplotlib transforms.
    The map uses one uniform pixel-per-meter scale, so ENU geometry is not
    stretched independently along the two axes.
    """

    if not positions:
        raise ValueError("Cannot plot an empty position set")
    destination = Path(output_path)
    ensure_parent(destination)
    by_id = {position.image_id: position for position in positions}

    width, height = 1800, 1400
    left, right, top, bottom = 150, 70, 120, 150
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    east = np.array([position.east for position in positions], dtype=np.float64)
    north = np.array([position.north for position in positions], dtype=np.float64)
    min_east, max_east = float(east.min()), float(east.max())
    min_north, max_north = float(north.min()), float(north.max())
    span_east = max(max_east - min_east, 1.0)
    span_north = max(max_north - min_north, 1.0)
    plot_width = width - left - right
    plot_height = height - top - bottom
    scale = min(plot_width / span_east, plot_height / span_north)
    used_width = span_east * scale
    used_height = span_north * scale
    offset_x = left + (plot_width - used_width) / 2.0
    offset_y = top + (plot_height - used_height) / 2.0

    def project(east_m: float, north_m: float) -> tuple[int, int]:
        x = offset_x + (east_m - min_east) * scale
        y = offset_y + (max_north - north_m) * scale
        return int(round(x)), int(round(y))

    font = cv2.FONT_HERSHEY_SIMPLEX
    grid_color = (225, 225, 225)
    axis_color = (165, 165, 165)
    label_color = (55, 55, 55)
    edge_color = (165, 150, 130)
    node_color = (178, 114, 0)

    for fraction in np.linspace(0.0, 1.0, 6):
        east_value = min_east + fraction * span_east
        north_value = min_north + fraction * span_north
        x, _ = project(east_value, min_north)
        _, y = project(min_east, north_value)
        cv2.line(canvas, (x, int(offset_y)), (x, int(offset_y + used_height)), grid_color, 1, cv2.LINE_AA)
        cv2.line(canvas, (int(offset_x), y), (int(offset_x + used_width), y), grid_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{east_value:.1f}", (x - 35, int(offset_y + used_height) + 34), font, 0.55, label_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{north_value:.1f}", (int(offset_x) - 100, y + 6), font, 0.55, label_color, 1, cv2.LINE_AA)

    if min_east <= 0.0 <= max_east:
        x0, _ = project(0.0, min_north)
        cv2.line(canvas, (x0, int(offset_y)), (x0, int(offset_y + used_height)), axis_color, 2, cv2.LINE_AA)
    if min_north <= 0.0 <= max_north:
        _, y0 = project(min_east, 0.0)
        cv2.line(canvas, (int(offset_x), y0), (int(offset_x + used_width), y0), axis_color, 2, cv2.LINE_AA)

    for first_id, second_id in pairs:
        first, second = by_id[first_id], by_id[second_id]
        cv2.line(
            canvas,
            project(first.east, first.north),
            project(second.east, second.north),
            edge_color,
            1,
            cv2.LINE_AA,
        )
    for position in positions:
        point = project(position.east, position.north)
        cv2.circle(canvas, point, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, point, 5, node_color, -1, cv2.LINE_AA)
        capture = f"#{position.capture_index:04d}" if position.capture_index is not None else "#?"
        cv2.putText(
            canvas,
            f"{position.image_id}/{capture}",
            (point[0] + 7, point[1] - 7),
            font,
            0.43,
            label_color,
            1,
            cv2.LINE_AA,
        )

    title = f"GPS Neighbor Graph - {method} ({len(positions)} images, {len(pairs)} pairs)"
    cv2.putText(canvas, title, (left, 55), font, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "East (m)", (width // 2 - 50, height - 42), font, 0.75, label_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "North (m)", (18, 85), font, 0.75, label_color, 2, cv2.LINE_AA)
    cv2.rectangle(
        canvas,
        (int(offset_x), int(offset_y)),
        (int(offset_x + used_width), int(offset_y + used_height)),
        (130, 130, 130),
        1,
    )
    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"OpenCV failed to write graph visualization: {destination}")


def draw_uniform_match_debug(
    image_i_path: str | Path,
    image_j_path: str | Path,
    inlier_pts_i: np.ndarray,
    inlier_pts_j: np.ndarray,
    selected_indices: np.ndarray,
    output_path: str | Path,
    *,
    grid_rows: int,
    grid_cols: int,
    max_height: int = 1200,
) -> None:
    """Draw overlap grid, all RANSAC inliers, and selected optimization matches."""

    first = cv2.imread(str(Path(image_i_path)), cv2.IMREAD_COLOR)
    second = cv2.imread(str(Path(image_j_path)), cv2.IMREAD_COLOR)
    if first is None or second is None:
        raise OSError("Could not decode one of the match-visualization images")
    scale = min(1.0, max_height / max(first.shape[0], second.shape[0]))
    if scale < 1.0:
        first = cv2.resize(first, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        second = cv2.resize(second, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    height = max(first.shape[0], second.shape[0])
    canvas = np.full((height, first.shape[1] + second.shape[1], 3), 245, dtype=np.uint8)
    canvas[: first.shape[0], : first.shape[1]] = first
    canvas[: second.shape[0], first.shape[1] :] = second
    points_i = np.asarray(inlier_pts_i, dtype=np.float64) * scale
    points_j = np.asarray(inlier_pts_j, dtype=np.float64) * scale
    points_j[:, 0] += first.shape[1]
    selected_set = set(np.asarray(selected_indices, dtype=np.int32).tolist())
    for index, (point_i, point_j) in enumerate(zip(points_i, points_j, strict=True)):
        start = tuple(np.rint(point_i).astype(int))
        end = tuple(np.rint(point_j).astype(int))
        if index in selected_set:
            cv2.line(canvas, start, end, (0, 80, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, start, 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, end, 5, (0, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.line(canvas, start, end, (140, 190, 140), 1, cv2.LINE_AA)
    if points_i.size:
        minimum = points_i.min(axis=0)
        maximum = points_i.max(axis=0)
        for column in range(grid_cols + 1):
            x = int(round(minimum[0] + (maximum[0] - minimum[0]) * column / grid_cols))
            cv2.line(canvas, (x, int(minimum[1])), (x, int(maximum[1])), (255, 180, 0), 1, cv2.LINE_AA)
        for row in range(grid_rows + 1):
            y = int(round(minimum[1] + (maximum[1] - minimum[1]) * row / grid_rows))
            cv2.line(canvas, (int(minimum[0]), y), (int(maximum[0]), y), (255, 180, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"RANSAC inliers={len(points_i)} selected={len(selected_set)}", (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2, cv2.LINE_AA)
    destination = Path(output_path)
    ensure_parent(destination)
    if not cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Could not write match visualization: {destination}")
