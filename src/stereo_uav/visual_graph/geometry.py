"""Geometry for the visual graph algorithm."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import replace

import numpy as np

from .models import (
    ImageEdge,
    ImageKey,
    ImagePair,
)


def image_order(key: ImageKey) -> tuple[int, int]:
    return key[1], 0 if key[0] == "left" else 1


def key_text(key: ImageKey) -> str:
    return f"{key[0]}:{key[1]}"


def canonical_image_pair(key_i: ImageKey, key_j: ImageKey) -> ImagePair:
    assert key_i != key_j
    return tuple(sorted((key_i, key_j)))


def signed_denominator(w: np.ndarray) -> np.ndarray:
    return np.where(np.abs(w) < 1e-10, np.where(w < 0, -1e-10, 1e-10), w)


def warp_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    xy = points @ H[:2, :2].T + H[:2, 2]
    w = signed_denominator(points @ H[2, :2] + H[2, 2])
    return xy / w[:, None]


def assert_edge_orientation(edge: ImageEdge) -> None:
    assert edge.key_i != edge.key_j
    assert edge.inlier_pts_i.shape == edge.inlier_pts_j.shape
    assert edge.inlier_pts_i.shape == (edge.ransac_inliers, 2)
    assert edge.selected_pts_i.shape == edge.selected_pts_j.shape
    assert edge.selected_pts_i.shape == (edge.selected_matches, 2)
    assert np.isfinite(edge.H_i_to_j).all()
    error = np.linalg.norm(
        warp_points(edge.H_i_to_j, edge.inlier_pts_i) - edge.inlier_pts_j, axis=1
    ).mean()
    assert np.isclose(error, edge.mean_reproj_error, rtol=1e-5, atol=1e-6), (
        "Endpoint/point/H orientation inconsistent",
        error,
        edge.mean_reproj_error,
    )


def reverse_edge(edge: ImageEdge) -> ImageEdge:
    H = np.linalg.inv(edge.H_i_to_j)
    H /= H[2, 2]
    error = float(
        np.linalg.norm(warp_points(H, edge.inlier_pts_j) - edge.inlier_pts_i, axis=1).mean()
    )
    quality = (
        min(edge.coverage_i, edge.coverage_j)
        * edge.inlier_ratio
        * math.log1p(edge.ransac_inliers)
        / (error + 1e-6)
    )
    result = replace(
        edge,
        key_i=edge.key_j,
        key_j=edge.key_i,
        inlier_pts_i=edge.inlier_pts_j,
        inlier_pts_j=edge.inlier_pts_i,
        selected_pts_i=edge.selected_pts_j,
        selected_pts_j=edge.selected_pts_i,
        coverage_i=edge.coverage_j,
        coverage_j=edge.coverage_i,
        H_i_to_j=H,
        mean_reproj_error=error,
        quality_score=quality,
        roles=set(edge.roles),
    )
    assert_edge_orientation(result)
    return result


def find_connected_components(
    nodes: list[ImageKey], edges: list[ImageEdge]
) -> list[list[ImageKey]]:
    adjacency: defaultdict = defaultdict(set)
    for edge in edges:
        adjacency[edge.key_i].add(edge.key_j)
        adjacency[edge.key_j].add(edge.key_i)
    remaining, components = set(nodes), []
    while remaining:
        start = min(remaining, key=image_order)
        remaining.remove(start)
        queue, component = deque([start]), []
        while queue:
            key = queue.popleft()
            component.append(key)
            for neighbor in sorted(adjacency[key] & remaining, key=image_order):
                remaining.remove(neighbor)
                queue.append(neighbor)
        components.append(sorted(component, key=image_order))
    return sorted(components, key=lambda group: (-len(group), image_order(group[0])))


def choose_reference(nodes: list[ImageKey]) -> ImageKey:
    left = [key for key in nodes if key[0] == "left"]
    return min(left or nodes, key=image_order)
