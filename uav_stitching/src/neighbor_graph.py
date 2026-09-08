"""Paper-default four-quadrant GPS neighbor graph and KNN baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .gps import ImagePosition


Pair = tuple[int, int]


@dataclass(frozen=True)
class GraphStatistics:
    """Basic diagnostics for a generated undirected image graph."""

    image_count: int
    pair_count: int
    connected_components: int
    isolated_images: tuple[int, ...]
    min_degree: int
    max_degree: int
    mean_degree: float


def quadrant_name(dx: float, dy: float) -> str:
    """Assign an offset to a deterministic half-open quadrant.

    The paper does not state how points exactly on an axis are assigned. This
    implementation choice uses north for ``dy == 0`` and east for ``dx == 0``:
    NE: dx>=0,dy>=0; NW: dx<0,dy>=0; SW: dx<0,dy<0; SE: dx>=0,dy<0.
    """

    if dx >= 0.0 and dy >= 0.0:
        return "NE"
    if dx < 0.0 <= dy:
        return "NW"
    if dx < 0.0 and dy < 0.0:
        return "SW"
    return "SE"


def select_quadrant_neighbors(
    current: ImagePosition,
    positions: list[ImagePosition],
    max_neighbor_distance_m: float | None = None,
) -> dict[str, int]:
    """Select at most the nearest image in each ENU quadrant around *current*."""

    if max_neighbor_distance_m is not None and max_neighbor_distance_m <= 0:
        raise ValueError("max_neighbor_distance_m must be positive or null")
    best: dict[str, tuple[float, int]] = {}
    for candidate in positions:
        if candidate.image_id == current.image_id:
            continue
        dx = candidate.east - current.east
        dy = candidate.north - current.north
        distance = math.hypot(dx, dy)
        if max_neighbor_distance_m is not None and distance > max_neighbor_distance_m:
            continue
        quadrant = quadrant_name(dx, dy)
        key = (distance, candidate.image_id)
        if quadrant not in best or key < best[quadrant]:
            best[quadrant] = key
    return {quadrant: image_id for quadrant, (_, image_id) in sorted(best.items())}


def build_quadrant_neighbor_graph(
    positions: list[ImagePosition],
    max_neighbor_distance_m: float | None = None,
) -> list[Pair]:
    """Build the paper's undirected four-quadrant GPS candidate-pair graph.

    Each image actively selects at most one nearest neighbor per quadrant. All
    directed selections are then normalized as ``(min_id, max_id)`` and
    deduplicated, so ``(i,j)`` and ``(j,i)`` are one matching pair.
    """

    _validate_positions(positions)
    pairs: set[Pair] = set()
    for current in positions:
        for neighbor_id in select_quadrant_neighbors(
            current,
            positions,
            max_neighbor_distance_m=max_neighbor_distance_m,
        ).values():
            pairs.add(tuple(sorted((current.image_id, neighbor_id))))
    return sorted(pairs)


def build_knn_neighbor_graph(
    positions: list[ImagePosition],
    k: int = 4,
    max_neighbor_distance_m: float | None = None,
) -> list[Pair]:
    """Build an undirected Euclidean KNN graph for ablation only."""

    _validate_positions(positions)
    if k <= 0:
        raise ValueError("k must be positive")
    if max_neighbor_distance_m is not None and max_neighbor_distance_m <= 0:
        raise ValueError("max_neighbor_distance_m must be positive or null")
    pairs: set[Pair] = set()
    for current in positions:
        candidates = []
        for candidate in positions:
            if candidate.image_id == current.image_id:
                continue
            distance = math.hypot(candidate.east - current.east, candidate.north - current.north)
            if max_neighbor_distance_m is None or distance <= max_neighbor_distance_m:
                candidates.append((distance, candidate.image_id))
        for _, neighbor_id in sorted(candidates)[:k]:
            pairs.add(tuple(sorted((current.image_id, neighbor_id))))
    return sorted(pairs)


def graph_statistics(positions: list[ImagePosition], pairs: list[Pair]) -> GraphStatistics:
    """Calculate connectivity and degree diagnostics without an optional graph package."""

    ids = {position.image_id for position in positions}
    adjacency = {image_id: set() for image_id in ids}
    for left, right in pairs:
        if left not in ids or right not in ids:
            raise ValueError(f"Pair ({left}, {right}) references an unknown image ID")
        adjacency[left].add(right)
        adjacency[right].add(left)

    remaining = set(ids)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            node = stack.pop()
            unseen = adjacency[node] & remaining
            remaining.difference_update(unseen)
            stack.extend(unseen)
    degrees = [len(adjacency[image_id]) for image_id in sorted(ids)]
    isolated = tuple(image_id for image_id in sorted(ids) if not adjacency[image_id])
    return GraphStatistics(
        image_count=len(ids),
        pair_count=len(pairs),
        connected_components=components,
        isolated_images=isolated,
        min_degree=min(degrees, default=0),
        max_degree=max(degrees, default=0),
        mean_degree=sum(degrees) / len(degrees) if degrees else 0.0,
    )


def _validate_positions(positions: list[ImagePosition]) -> None:
    ids = [position.image_id for position in positions]
    if len(ids) != len(set(ids)):
        raise ValueError("Image IDs must be unique")
    for position in positions:
        if not math.isfinite(position.east) or not math.isfinite(position.north):
            raise ValueError(f"Image {position.image_id} has a non-finite ENU position")
