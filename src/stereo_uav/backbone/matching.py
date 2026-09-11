"""Matching for the backbone algorithm."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from . import config
from .io import (
    image_pixels,
)
from .models import (
    EdgeSpec,
    FeatureRecord,
    ImageRecord,
    PairMatch_Edge,
)


def extract_features(
    images: Sequence[ImageRecord],
    cache: Optional[Dict[Path, FeatureRecord]] = None,
    sift=None,
) -> List[FeatureRecord]:
    """Extract SIFT keypoints and descriptors from grayscale images."""

    print("Extracting SIFT features")

    if sift is None:
        sift = cv2.SIFT_create(nfeatures=config.SIFT_NFEATURES)

    features: List[FeatureRecord] = []  # Feature records.

    for image in tqdm(images, desc="SIFT"):
        if cache is not None and image.path in cache:
            features.append(cache[image.path])
            continue

        if image.gray is None:
            pixels = image_pixels(image)
            gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.gray

        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) == 0:
            print(f"Warning: no SIFT features in image {image.index}: {image.name}")
            keypoints = []
            descriptors = None

        feature = FeatureRecord(keypoints, descriptors)
        features.append(feature)
        if cache is not None:
            cache[image.path] = feature
    return features


# Build candidate matching edges.
def build_candidate_edges(num_images: int) -> List[EdgeSpec]:
    """Return six typed edges for [left_t, left_t1, right_t, right_t1]. Edges (0, 2) and (1, 3) are

    stereo; the other four are ordinary. Preserve these types throughout matching and
    optimization.
    """
    if num_images != 4:
        raise ValueError(
            "Special-edge block requires exactly 4 images ordered as "
            "[left_t, left_t+1, right_t, right_t+1], "
            f"but got num_images={num_images}."
        )

    edges = list(config.BLOCK_EDGE_SPECS)
    print("Built 6 typed candidate edges: 2 stereo special edges + 4 ordinary edges")
    return edges


def build_candidate_pairs(num_images: int) -> List[Tuple[int, int]]:
    """Return endpoint pairs for legacy callers; the main pipeline uses typed edges."""
    return [(spec.i, spec.j) for spec in build_candidate_edges(num_images)]


def compute_reprojection_error(H_i_to_j: np.ndarray, pts_i: np.ndarray, pts_j: np.ndarray) -> float:
    """Map pts_i into image j and return the mean Euclidean distance to pts_j."""

    if len(pts_i) == 0:
        return math.inf
    # Input points have shape (N, 2).

    projected = cv2.perspectiveTransform(pts_i.reshape(-1, 1, 2).astype(np.float64), H_i_to_j)
    # OpenCV perspectiveTransform expects shape (N, 1, 2).
    # Infer N while preserving the coordinate dimension.

    # Restore the (N, 2) point layout.
    projected = projected.reshape(-1, 2)

    # Compute the Euclidean reprojection error per correspondence.
    errors = np.linalg.norm(projected - pts_j, axis=1)
    return float(np.mean(errors))  # Return the mean pixel error.


# Select spatial grid representatives and fill with farthest-point sampling.


def select_uniform_matches(
    inlier_matches: Sequence[cv2.DMatch],
    keypoints_i: Sequence[cv2.KeyPoint],
    keypoints_j: Sequence[cv2.KeyPoint],
    target_count: int = config.SELECTED_MATCHES_PER_PAIR,
) -> Tuple[List[cv2.DMatch], np.ndarray, np.ndarray]:
    """Sample spatially distributed RANSAC inliers. Select a strong match per occupied source

    bounding-box grid cell, then fill using farthest-point sampling instead of taking descriptor
    Top-P alone.
    """

    if not inlier_matches:
        return [], np.empty((0, 2), np.float64), np.empty((0, 2), np.float64)

    pts_i = np.array([keypoints_i[m.queryIdx].pt for m in inlier_matches], dtype=np.float64)
    pts_j = np.array([keypoints_j[m.trainIdx].pt for m in inlier_matches], dtype=np.float64)
    distances = np.array([m.distance for m in inlier_matches], dtype=np.float64)

    x0, y0 = np.min(pts_i, axis=0)
    x1, y1 = np.max(pts_i, axis=0)
    if x1 <= x0 or y1 <= y0:
        order = np.argsort(distances)[:target_count]
        selected = [inlier_matches[int(idx)] for idx in order]
        return selected, pts_i[order], pts_j[order]

    selected_indices: List[int] = []
    used = set()
    for row in range(config.GRID_ROWS):
        for col in range(config.GRID_COLS):
            cx0 = x0 + (x1 - x0) * col / config.GRID_COLS
            cx1 = x0 + (x1 - x0) * (col + 1) / config.GRID_COLS
            cy0 = y0 + (y1 - y0) * row / config.GRID_ROWS
            cy1 = y0 + (y1 - y0) * (row + 1) / config.GRID_ROWS
            in_x = (pts_i[:, 0] >= cx0) & (
                pts_i[:, 0] <= cx1 if col == config.GRID_COLS - 1 else pts_i[:, 0] < cx1
            )
            in_y = (pts_i[:, 1] >= cy0) & (
                pts_i[:, 1] <= cy1 if row == config.GRID_ROWS - 1 else pts_i[:, 1] < cy1
            )
            cell_indices = np.where(in_x & in_y)[0]
            if len(cell_indices) == 0:
                continue
            best = int(cell_indices[np.argmin(distances[cell_indices])])
            if best not in used:
                selected_indices.append(best)
                used.add(best)

    remaining = [idx for idx in range(len(inlier_matches)) if idx not in used]
    if len(selected_indices) < target_count and remaining:
        norm = pts_i.copy()
        norm[:, 0] = (norm[:, 0] - x0) / max(x1 - x0, 1e-9)
        norm[:, 1] = (norm[:, 1] - y0) / max(y1 - y0, 1e-9)

        if not selected_indices:
            first = int(remaining[int(np.argmin(distances[remaining]))])
            selected_indices.append(first)
            used.add(first)
            remaining = [idx for idx in remaining if idx != first]

        while len(selected_indices) < target_count and remaining:
            selected_pts = norm[np.array(selected_indices)]
            rem_pts = norm[np.array(remaining)]
            min_d2 = np.min(
                np.sum((rem_pts[:, None, :] - selected_pts[None, :, :]) ** 2, axis=2), axis=1
            )
            candidate_order = np.lexsort((distances[remaining], -min_d2))
            best_remaining_pos = int(candidate_order[0])
            best_idx = int(remaining[best_remaining_pos])
            selected_indices.append(best_idx)
            used.add(best_idx)
            remaining.pop(best_remaining_pos)

    selected_indices = selected_indices[: min(target_count, len(selected_indices))]
    selected_indices_np = np.array(selected_indices, dtype=np.int64)
    selected = [inlier_matches[int(idx)] for idx in selected_indices_np]
    return selected, pts_i[selected_indices_np], pts_j[selected_indices_np]


def match_pair(spec: EdgeSpec, features: Sequence[FeatureRecord]) -> PairMatch_Edge:
    """Match one edge using BFMatcher L2 KNN with k=2, Lowe ratio filtering, homography RANSAC, and

    spatial sampling when enough inliers survive.
    """

    i, j = spec.i, spec.j
    pair = PairMatch_Edge(
        i=i,
        j=j,
        edge_type=spec.edge_type,
        edge_name=spec.edge_name,
    )

    fi = features[i]
    fj = features[j]

    if fi.descriptors is None or fj.descriptors is None:
        return pair

    if len(fi.descriptors) < 2 or len(fj.descriptors) < 2:
        return pair

    # Find the two nearest descriptors using BFMatcher with L2 distance.
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = matcher.knnMatch(fi.descriptors, fj.descriptors, k=2)

    pair.raw_matches = len(knn)

    # Reject ambiguous descriptor matches with the Lowe ratio test.

    good: List[cv2.DMatch] = []
    for candidates in knn:
        if len(candidates) != 2:
            continue
        m, n = candidates
        if m.distance < config.RATIO_TEST * n.distance:
            good.append(m)
    pair.ratio_matches = len(good)
    if len(good) < 4:
        return pair

    # Collect retained source keypoint coordinates.
    pts_i = np.float64([fi.keypoints[m.queryIdx].pt for m in good])

    # Collect their target keypoint coordinates.
    pts_j = np.float64([fj.keypoints[m.trainIdx].pt for m in good])

    # Estimate a homography with RANSAC.
    H, mask = cv2.findHomography(pts_i, pts_j, cv2.RANSAC, config.RANSAC_REPROJ_THRESH)

    if H is None or mask is None:
        return pair

    mask = mask.reshape(-1).astype(bool)

    # Count RANSAC inliers.
    inlier_matches = [m for m, keep in zip(good, mask) if keep]

    # Normalize the homography's homogeneous scale.
    pair.H_i_to_j = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H

    pair.ransac_inliers = len(inlier_matches)

    pair.inlier_ratio = float(pair.ransac_inliers / max(pair.ratio_matches, 1))

    if pair.ransac_inliers > 0:
        inlier_pts_i = np.float64([fi.keypoints[m.queryIdx].pt for m in inlier_matches])
        inlier_pts_j = np.float64([fj.keypoints[m.trainIdx].pt for m in inlier_matches])
        pair.mean_reproj_error = compute_reprojection_error(
            pair.H_i_to_j, inlier_pts_i, inlier_pts_j
        )

    if pair.ransac_inliers < config.MIN_INLIERS:
        return pair

    # Distribute selected inliers spatially with grid and farthest-point sampling.

    selected, selected_pts_i, selected_pts_j = select_uniform_matches(
        inlier_matches,
        fi.keypoints,
        fj.keypoints,
        config.SELECTED_MATCHES_PER_PAIR,
    )

    min_selected = min(config.MIN_INLIERS, config.SELECTED_MATCHES_PER_PAIR)

    if len(selected) < min_selected:
        pair.selected_matches = len(selected)
        return pair

    pair.inlier_matches = inlier_matches
    pair.selected_dmatches = selected
    pair.selected_pts_i = selected_pts_i
    pair.selected_pts_j = selected_pts_j
    pair.selected_matches = len(selected)

    return pair


def match_block_edges(features: Sequence[FeatureRecord]) -> List[PairMatch_Edge]:
    """Match the fixed typed graph; the main loop processes individual edges to reuse cached

    observations.
    """
    return [match_pair(spec, features) for spec in build_candidate_edges(len(features))]


def filter_valid_edges(
    pair_results: Sequence[PairMatch_Edge],
    num_images: int,
) -> List[PairMatch_Edge]:
    """Filter valid edges and require every block image to connect to the reference image."""

    valid = [
        edge
        for edge in pair_results
        if edge.selected_matches >= min(config.MIN_INLIERS, config.SELECTED_MATCHES_PER_PAIR)
    ]
    print(f"  Valid registration edges: {len(valid)} / {len(pair_results)}")
    if not valid:
        raise RuntimeError("No valid image pair survived matching/RANSAC/uniform selection.")
    if sum(edge.selected_matches for edge in valid) < 4 * max(1, len(valid)):
        raise RuntimeError("Too few selected matches for stable optimization.")

    adjacency = {idx: set() for idx in range(num_images)}
    for edge in valid:
        adjacency[edge.i].add(edge.j)
        adjacency[edge.j].add(edge.i)

    visited = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        for neighbor in adjacency[current] - visited:
            visited.add(neighbor)
            pending.append(neighbor)

    if len(visited) != num_images:
        missing = sorted(set(range(num_images)) - visited)
        raise RuntimeError(
            f"Registration graph is disconnected; images {missing} cannot be aligned to reference image 0."
        )
    return valid
