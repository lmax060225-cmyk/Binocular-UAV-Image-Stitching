"""Graph for the visual graph algorithm."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, replace
from itertools import combinations
from typing import Callable

from .geometry import (
    assert_edge_orientation,
    canonical_image_pair,
    find_connected_components,
)
from .models import (
    Config,
    GraphBuildResult,
    ImageEdge,
    ImageKey,
    ImagePair,
    PairRelation,
    StereoPair,
)
from .retrieval import (
    IncrementalSIFTRetriever,
)


def relation_quality(edges: list[ImageEdge], cfg: Config) -> tuple[float, int]:
    best = sorted(edges, key=lambda edge: -edge.quality_score)[:2]
    endpoints = {key for edge in best for key in (edge.key_i, edge.key_j)}
    score = sum(
        edge.quality_score * (1.0 if index == 0 else 0.5) for index, edge in enumerate(best)
    )
    return score + cfg.ENDPOINT_COVERAGE_BONUS * len(endpoints), len(endpoints)


def verify_pair_relation(
    current: StereoPair, historical: StereoPair, source: str, verify: Callable, cfg: Config
) -> PairRelation | None:
    # Always attempt all four LL/LR/RL/RR combinations, including cross-camera.
    edges = []
    for ki in (current.left_key, current.right_key):
        for kj in (historical.left_key, historical.right_key):
            edge = verify(ki, kj)
            if edge is not None:
                assert {edge.key_i, edge.key_j} == {ki, kj}
                edges.append(edge)
    if not edges:
        return None
    score, coverage = relation_quality(edges, cfg)
    return PairRelation(current.frame, historical.frame, source, edges, score, coverage)


def select_relation_image_edges(
    relation: PairRelation,
    role: str,
    new_pair: StereoPair,
    historical_nodes: set[ImageKey],
    intra_edge: ImageEdge | None = None,
) -> list[ImageEdge]:
    edges = relation.verified_image_edges
    if not edges:
        return []
    if role == "loop":
        ordered = sorted(edges, key=lambda e: -e.quality_score)
        best = ordered[:1]
        endpoints = {best[0].key_i, best[0].key_j}
        for edge in ordered[1:]:
            if edge.quality_score >= 0.8 * best[0].quality_score and len(
                endpoints | {edge.key_i, edge.key_j}
            ) > len(endpoints):
                best.append(edge)
                break
        return best
    new_nodes = {new_pair.left_key, new_pair.right_key}

    def rank(subset):
        endpoints = {key for e in subset for key in (e.key_i, e.key_j)}
        connected_new = set()
        for edge in subset:
            if edge.key_i in historical_nodes:
                connected_new.add(edge.key_j)
            if edge.key_j in historical_nodes:
                connected_new.add(edge.key_i)
        connected_new &= new_nodes
        if connected_new and intra_edge is not None:
            connected_new = new_nodes
        secondary = (len(endpoints), sum(e.quality_score for e in subset))
        return (len(connected_new), *secondary) if role == "tree" else secondary

    subsets = [subset for count in (1, 2) for subset in combinations(edges, count)]
    return list(max(subsets, key=rank))


def strict_loop_edges(relation: PairRelation, cfg: Config) -> list[ImageEdge]:
    return [
        e
        for e in relation.verified_image_edges
        if e.ransac_inliers >= cfg.LOOP_MIN_RANSAC_INLIERS
        and e.inlier_ratio >= cfg.LOOP_MIN_INLIER_RATIO
        and e.mean_reproj_error <= cfg.LOOP_MAX_REPROJ_ERROR
        and min(e.coverage_i, e.coverage_j) >= cfg.LOOP_MIN_COVERAGE
    ]


def cull_strong_relations(relations: dict[tuple[int, int], PairRelation], cfg: Config) -> int:
    removed = 0
    while True:
        degree: Counter = Counter()
        for relation in relations.values():
            if "strong" in relation.roles:
                degree.update((relation.frame_i, relation.frame_j))
        overloaded = {
            frame for frame, value in degree.items() if value > cfg.MAX_STRONG_PAIR_DEGREE
        }
        if not overloaded:
            return removed
        choices = [
            (key, relation)
            for key, relation in relations.items()
            if "strong" in relation.roles and overloaded & {relation.frame_i, relation.frame_j}
        ]
        key, weakest = min(choices, key=lambda item: (item[1].relation_score, item[0]))
        weakest.roles.remove("strong")
        weakest.role_edges.pop("strong", None)
        if not weakest.roles:
            del relations[key]
        removed += 1


def materialize_optimization_graph(
    intra_edges: list[ImageEdge], relations: dict[tuple[int, int], PairRelation]
) -> dict[ImagePair, ImageEdge]:
    result: dict[ImagePair, ImageEdge] = {}

    def add(edge: ImageEdge, role: str):
        assert_edge_orientation(edge)
        identity = canonical_image_pair(edge.key_i, edge.key_j)
        previous = result.get(identity)
        roles = {role} | (previous.roles if previous else set())
        best = edge if previous is None or edge.quality_score > previous.quality_score else previous
        provenance = next(r for r in ("intra_pair", "tree", "loop", "strong") if r in roles)
        result[identity] = replace(best, provenance=provenance, roles=roles)

    for edge in intra_edges:
        add(edge, "intra_pair")
    for relation in relations.values():
        distinct_pairs = {
            canonical_image_pair(edge.key_i, edge.key_j)
            for role in relation.roles
            for edge in relation.role_edges[role]
        }
        assert len(distinct_pairs) <= 2, "A pair relation may materialize at most two image edges"
        for role in sorted(relation.roles):
            selected = relation.role_edges[role]
            assert 1 <= len(selected) <= 2
            for edge in selected:
                assert any(edge is candidate for candidate in relation.verified_image_edges)
                add(edge, role)
    return result


def build_incremental_pair_graph(
    pairs: list[StereoPair], verifier, cfg: Config, retriever=None
) -> GraphBuildResult:
    retriever = retriever or IncrementalSIFTRetriever(cfg)
    relations, intra_edges, retrieval_rows, status_rows = {}, [], [], []
    counts = dict(sequential=0, retrieval=0, verified=0, pruned_strong=0)
    by_frame = {pair.frame: pair for pair in pairs}
    historical_nodes: set[ImageKey] = set()
    started = time.perf_counter()
    for position, pair in enumerate(pairs):
        intra = verifier.verify(pair.left_key, pair.right_key)
        if intra is not None:
            intra_edges.append(intra)
        descriptors = retriever.describe_pair(pair, verifier.feature)
        retrieved = retriever.query(descriptors, position)
        candidates: dict[int, str] = {}
        for offset in cfg.SEQUENTIAL_OFFSETS:
            if position >= offset:
                candidates[pairs[position - offset].frame] = "sequential"
                counts["sequential"] += 1
        for candidate in retrieved:
            assert candidate.position < position
            assert position - candidate.position >= cfg.RETRIEVAL_TEMPORAL_EXCLUSION
            retrieval_rows.append(
                dict(query_frame=pair.frame, query_position=position, **asdict(candidate))
            )
            counts["retrieval"] += 1
            candidates.setdefault(candidate.frame, "retrieval")
        verified = []
        for frame, source in candidates.items():
            relation = verify_pair_relation(pair, by_frame[frame], source, verifier.verify, cfg)
            if relation is not None:
                verified.append(relation)
        counts["verified"] += len(verified)
        sequential = [r for r in verified if r.source == "sequential"]
        parent = max(sequential or verified, key=lambda r: r.relation_score, default=None)

        def attach(relation, role, eligible=None):
            identity = tuple(sorted((relation.frame_i, relation.frame_j)))
            stored = relations.setdefault(identity, relation)
            material = (
                relation if eligible is None else replace(relation, verified_image_edges=eligible)
            )
            selected = select_relation_image_edges(material, role, pair, historical_nodes, intra)
            if selected:
                stored.roles.add(role)
                stored.role_edges[role] = selected

        if parent is not None:
            attach(parent, "tree")
        loops = []
        for relation in verified:
            # The parent already has permanent tree retention and its chosen
            # observations must never be replaced by an independent loop subset.
            if relation.source == "retrieval" and relation is not parent:
                eligible = strict_loop_edges(relation, cfg)
                if eligible:
                    score, _ = relation_quality(eligible, cfg)
                    loops.append((score, relation, eligible))
        selected_loops = sorted(loops, key=lambda item: -item[0])[: cfg.MAX_LOOP_RELATIONS]
        for _, relation, eligible in selected_loops:
            attach(relation, "loop", eligible)
        loop_ids = {id(r) for _, r, _ in selected_loops}
        strong = [r for r in verified if r is not parent and id(r) not in loop_ids]
        for relation in sorted(strong, key=lambda r: -r.relation_score)[
            : cfg.MAX_NEW_STRONG_RELATIONS
        ]:
            attach(relation, "strong")
        counts["pruned_strong"] += cull_strong_relations(relations, cfg)
        current_nodes = historical_nodes | {pair.left_key, pair.right_key}
        current_edges = list(materialize_optimization_graph(intra_edges, relations).values())
        components = find_connected_components(list(current_nodes), current_edges)
        reachable = next(group for group in components if pairs[0].left_key in group)
        registered = {pair.left_key, pair.right_key} <= set(reachable)
        if not registered:
            print(
                f"WARNING: frame {pair.frame}: registration deferred / disconnected pair",
                flush=True,
            )
        status_rows.append(
            dict(
                frame=pair.frame,
                position=position,
                parent_frame=parent.frame_j if parent else None,
                registered_at_arrival=registered,
                intra_verified=intra is not None,
                components_at_arrival=len(components),
            )
        )
        retriever.add(pair.frame, position, descriptors)
        historical_nodes = current_nodes
        print(
            f"Pair {position + 1}/{len(pairs)} frame={pair.frame}: "
            f"candidates={len(candidates)}, verified={len(verified)}, "
            f"parent={parent.frame_j if parent else '-'}, "
            f"image_edges={len(current_edges)}, elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    final = materialize_optimization_graph(intra_edges, relations)
    return GraphBuildResult(relations, intra_edges, final, counts, retrieval_rows, status_rows)
