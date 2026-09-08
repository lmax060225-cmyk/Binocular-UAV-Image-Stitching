"""Independent incremental visual graph and paper global registration.

Run with --self-test, or with no arguments for ./data_set_5.
Only NumPy, OpenCV, SciPy, pandas and the Python standard library are required.
All coordinates and reported errors are in original input-image pixels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
import unittest
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import lsmr


ImageKey = tuple[str, int]
ImagePair = tuple[ImageKey, ImageKey]
BASE_DIR = Path(__file__).resolve().parent
LEFT_IN = BASE_DIR / "data_set_5" / "left"
RIGHT_IN = BASE_DIR / "data_set_5" / "right"
OUTPUT_DIR = BASE_DIR / "out_new_visual_graph"


@dataclass(frozen=True)
class Config:
    FRAME_ID_REGEX: str = r"^(\d+)"
    SIFT_NFEATURES: int = 10000
    LOWE_RATIO: float = 0.75
    RANSAC_REPROJ_THRESH: float = 4.0
    MIN_RANSAC_INLIERS: int = 40
    MIN_INLIER_RATIO: float = 0.25
    MAX_MEAN_REPROJ_ERROR: float = 3.0
    MIN_SPATIAL_COVERAGE: float = 0.03
    RANDOM_SEED: int = 7
    SEQUENTIAL_OFFSETS: tuple[int, ...] = (1, 2, 4)
    RETRIEVAL_DESCRIPTORS_PER_IMAGE: int = 500
    RETRIEVAL_RATIO: float = 0.8
    RETRIEVAL_TOP_K: int = 20
    RETRIEVAL_TEMPORAL_EXCLUSION: int = 10
    ENDPOINT_COVERAGE_BONUS: float = 1.0
    MAX_NEW_STRONG_RELATIONS: int = 2
    MAX_STRONG_PAIR_DEGREE: int = 4
    LOOP_MIN_RANSAC_INLIERS: int = 50
    LOOP_MIN_INLIER_RATIO: float = 0.35
    LOOP_MAX_REPROJ_ERROR: float = 2.5
    LOOP_MIN_COVERAGE: float = 0.05
    MAX_LOOP_RELATIONS: int = 2
    OMEGA_RIGID: float = 800.0
    SIGMA_TR: float = 5000.0
    TRANSLATION_SCALE: float = 5000.0
    MAX_NFEV: int = 500
    FLANN_TREES: int = 4
    FLANN_CHECKS: int = 64
    # Rendering resource limits do not enter registration or alter saved H.
    MOSAIC_MAX_PIXELS: int = 30000000
    MOSAIC_MAX_SIDE: int = 16000
    GRAPHCUT_MAX_PIXELS: int = 250000


@dataclass(frozen=True)
class StereoPair:
    frame: int
    left_key: ImageKey
    right_key: ImageKey


@dataclass
class FeatureRecord:
    keypoints: np.ndarray
    descriptors: np.ndarray
    responses: np.ndarray
    shape: tuple[int, int]


@dataclass
class ImageEdge:
    key_i: ImageKey
    key_j: ImageKey
    provenance: str
    raw_matches: int
    ratio_matches: int
    ransac_inliers: int
    inlier_ratio: float
    mean_reproj_error: float
    H_i_to_j: np.ndarray
    inlier_pts_i: np.ndarray
    inlier_pts_j: np.ndarray
    selected_pts_i: np.ndarray
    selected_pts_j: np.ndarray
    selected_matches: int
    coverage_i: float
    coverage_j: float
    quality_score: float
    roles: set[str] = field(default_factory=set)


@dataclass
class PairRelation:
    frame_i: int
    frame_j: int
    source: str
    verified_image_edges: list[ImageEdge]
    relation_score: float
    endpoint_coverage: int
    roles: set[str] = field(default_factory=set)
    role_edges: dict[str, list[ImageEdge]] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalCandidate:
    frame: int
    position: int
    votes: int
    score: float
    weighted_score: float


@dataclass
class GraphBuildResult:
    relations: dict[tuple[int, int], PairRelation]
    intra_edges: list[ImageEdge]
    optimization_edges_by_pair: dict[ImagePair, ImageEdge]
    candidate_counts: dict[str, int]
    retrieval_rows: list[dict]
    pair_status_rows: list[dict]


def image_order(key: ImageKey) -> tuple[int, int]:
    return key[1], 0 if key[0] == "left" else 1


def key_text(key: ImageKey) -> str:
    return f"{key[0]}:{key[1]}"


def canonical_image_pair(key_i: ImageKey, key_j: ImageKey) -> ImagePair:
    assert key_i != key_j
    return tuple(sorted((key_i, key_j)))


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8),
                         cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode: {path}")
    data.tofile(path)


def load_stereo_pairs(left: Path, right: Path, cfg: Config
                      ) -> tuple[list[StereoPair], dict[ImageKey, Path]]:
    pattern = re.compile(cfg.FRAME_ID_REGEX)
    accepted = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    sides: dict[str, dict[int, Path]] = {}
    for camera, directory in (("left", left), ("right", right)):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        frames: dict[int, Path] = {}
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in accepted:
                continue
            match = pattern.match(path.name)
            if match is None:
                print(f"WARNING: no leading frame id; skipped {path}", flush=True)
                continue
            frame = int(match.group(1))
            if frame in frames:
                raise ValueError(f"Ambiguous duplicate {camera} frame {frame}: "
                                 f"{frames[frame].name}, {path.name}")
            frames[frame] = path
        sides[camera] = frames
    for frame in sorted(set(sides["left"]) ^ set(sides["right"])):
        missing = "left" if frame not in sides["left"] else "right"
        print(f"WARNING: frame {frame} missing {missing}; skipped", flush=True)
    pairs, paths = [], {}
    for frame in sorted(set(sides["left"]) & set(sides["right"])):
        pair = StereoPair(frame, ("left", frame), ("right", frame))
        pairs.append(pair)
        for key in (pair.left_key, pair.right_key):
            paths[key] = sides[key[0]][frame]
    if not pairs:
        raise ValueError("No synchronized pairs")
    return pairs, paths


def extract_features(key: ImageKey, paths: dict[ImageKey, Path],
                     feature_cache: dict[ImageKey, FeatureRecord],
                     sift, extraction_counts: Counter) -> FeatureRecord:
    if key not in feature_cache:
        gray = read_image(paths[key], grayscale=True)
        points, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None:
            descriptors = np.empty((0, 128), dtype=np.float32)
        feature_cache[key] = FeatureRecord(
            np.asarray([p.pt for p in points], dtype=np.float64).reshape(-1, 2),
            np.ascontiguousarray(descriptors, dtype=np.float32),
            np.asarray([p.response for p in points]), gray.shape)
        extraction_counts[key] += 1
        assert extraction_counts[key] == 1, "SIFT extracted more than once"
    return feature_cache[key]


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
    error = np.linalg.norm(warp_points(edge.H_i_to_j, edge.inlier_pts_i)
                           - edge.inlier_pts_j, axis=1).mean()
    assert np.isclose(error, edge.mean_reproj_error, rtol=1e-5, atol=1e-6), (
        "Endpoint/point/H orientation inconsistent", error, edge.mean_reproj_error)


def reverse_edge(edge: ImageEdge) -> ImageEdge:
    H = np.linalg.inv(edge.H_i_to_j)
    H /= H[2, 2]
    error = float(np.linalg.norm(warp_points(H, edge.inlier_pts_j)
                                 - edge.inlier_pts_i, axis=1).mean())
    quality = (min(edge.coverage_i, edge.coverage_j) * edge.inlier_ratio
               * math.log1p(edge.ransac_inliers) / (error + 1e-6))
    result = replace(edge, key_i=edge.key_j, key_j=edge.key_i,
                     inlier_pts_i=edge.inlier_pts_j, inlier_pts_j=edge.inlier_pts_i,
                     selected_pts_i=edge.selected_pts_j,
                     selected_pts_j=edge.selected_pts_i,
                     coverage_i=edge.coverage_j, coverage_j=edge.coverage_i,
                     H_i_to_j=H, mean_reproj_error=error, quality_score=quality,
                     roles=set(edge.roles))
    assert_edge_orientation(result)
    return result


def point_policy(total_images: int) -> tuple[int, int, int]:
    return (40, 5, 8) if total_images <= 300 else (20, 5, 4)


def select_evenly_distributed_matches(points_i: np.ndarray, points_j: np.ndarray,
                                      P: int, rows: int, cols: int,
                                      rng: np.random.Generator
                                      ) -> tuple[np.ndarray, np.ndarray]:
    if len(points_i) < P:
        raise ValueError(f"Need {P} inliers, received {len(points_i)}")
    assert rows * cols == P and points_i.shape == points_j.shape
    low, high = points_i.min(axis=0), points_i.max(axis=0)
    normalized = (points_i - low) / np.maximum(high - low, 1e-12)
    cell_x = np.minimum((normalized[:, 0] * cols).astype(int), cols - 1)
    cell_y = np.minimum((normalized[:, 1] * rows).astype(int), rows - 1)
    cells = cell_y * cols + cell_x
    selected = [int(rng.choice(np.flatnonzero(cells == cell)))
                for cell in np.unique(cells)]
    remaining = np.setdiff1d(np.arange(len(points_i)), selected)
    if len(selected) < P:
        selected.extend(rng.choice(remaining, P - len(selected), replace=False).tolist())
    assert len(selected) == len(set(selected)) == P
    return points_i[selected].copy(), points_j[selected].copy()


def spatial_coverage(points: np.ndarray, shape: tuple[int, int]) -> float:
    return float(cv2.contourArea(cv2.convexHull(points.astype(np.float32)))
                 / (shape[0] * shape[1]))


def mutual_ratio_matches(forward, backward, ratio: float):
    f = {m.queryIdx: m.trainIdx for pair in forward if len(pair) == 2
         for m, n in [pair] if m.distance < ratio * n.distance}
    b = {m.queryIdx: m.trainIdx for pair in backward if len(pair) == 2
         for m, n in [pair] if m.distance < ratio * n.distance}
    return f, [(i, j) for i, j in f.items() if b.get(j) == i]


class VisualVerifier:
    """Global SIFT and FLANN caches; failed and valid observations are cached too."""
    def __init__(self, paths: dict[ImageKey, Path], total_images: int, cfg: Config):
        self.paths, self.cfg = paths, cfg
        self.P, self.rows, self.cols = point_policy(total_images)
        self.feature_cache: dict[ImageKey, FeatureRecord] = {}
        self.extraction_counts: Counter = Counter()
        self.sift = cv2.SIFT_create(nfeatures=cfg.SIFT_NFEATURES)
        self.matchers: dict[ImageKey, object] = {}
        self.observations: dict[ImagePair, ImageEdge | None] = {}
        self.diagnostics: list[dict] = []

    def feature(self, key: ImageKey) -> FeatureRecord:
        return extract_features(key, self.paths, self.feature_cache,
                                self.sift, self.extraction_counts)

    def matcher(self, key: ImageKey):
        if key not in self.matchers:
            matcher = cv2.FlannBasedMatcher(
                dict(algorithm=1, trees=self.cfg.FLANN_TREES),
                dict(checks=self.cfg.FLANN_CHECKS))
            matcher.add([self.feature(key).descriptors])
            matcher.train()
            self.matchers[key] = matcher
        return self.matchers[key]

    def verify(self, key_i: ImageKey, key_j: ImageKey) -> ImageEdge | None:
        return verify_image_pair(key_i, key_j, self)


def verify_image_pair(key_i: ImageKey, key_j: ImageKey,
                      verifier: VisualVerifier) -> ImageEdge | None:
    identity = canonical_image_pair(key_i, key_j)
    if identity in verifier.observations:
        # Canonical identity never reorients the actual observation.
        return verifier.observations[identity]
    cfg = verifier.cfg
    fi, fj = verifier.feature(key_i), verifier.feature(key_j)
    row = dict(key_i=key_text(key_i), key_j=key_text(key_j),
               raw_matches=0, ratio_matches=0, mutual_matches=0,
               ransac_inliers=0, inlier_ratio=0., mean_reproj_error=np.nan,
               coverage_i=0., coverage_j=0., selected_matches=0,
               verified=False, reason="insufficient_descriptors")
    verifier.observations[identity] = None
    verifier.diagnostics.append(row)
    if min(len(fi.descriptors), len(fj.descriptors)) < 2:
        return None
    forward = verifier.matcher(key_j).knnMatch(fi.descriptors, k=2)
    backward = verifier.matcher(key_i).knnMatch(fj.descriptors, k=2)
    ratio, mutual = mutual_ratio_matches(forward, backward, cfg.LOWE_RATIO)
    row.update(raw_matches=len(forward), ratio_matches=len(ratio),
               mutual_matches=len(mutual), reason="insufficient_mutual_matches")
    if len(mutual) < max(4, verifier.P, cfg.MIN_RANSAC_INLIERS):
        return None
    indices = np.asarray(mutual)
    pi, pj = fi.keypoints[indices[:, 0]], fj.keypoints[indices[:, 1]]
    H, mask = cv2.findHomography(pi, pj, cv2.RANSAC, cfg.RANSAC_REPROJ_THRESH,
                                maxIters=4000, confidence=0.995)
    row["reason"] = "homography_failed"
    if H is None or mask is None or not np.isfinite(H).all():
        return None
    if abs(H[2, 2]) < 1e-12:
        return None
    H = H / H[2, 2]
    mask = mask.ravel().astype(bool)
    pi, pj = pi[mask], pj[mask]
    count, inlier_ratio = len(pi), float(mask.mean())
    if count < 3:
        return None
    error = float(np.linalg.norm(warp_points(H, pi) - pj, axis=1).mean())
    ci, cj = spatial_coverage(pi, fi.shape), spatial_coverage(pj, fj.shape)
    row.update(ransac_inliers=count, inlier_ratio=inlier_ratio,
               mean_reproj_error=error, coverage_i=ci, coverage_j=cj,
               reason="quality_gate_failed")
    failures = []
    if count < max(cfg.MIN_RANSAC_INLIERS, verifier.P):
        failures.append("inlier_count")
    if inlier_ratio < cfg.MIN_INLIER_RATIO:
        failures.append("inlier_ratio")
    if not np.isfinite(error) or error > cfg.MAX_MEAN_REPROJ_ERROR:
        failures.append("reprojection")
    if min(ci, cj) < cfg.MIN_SPATIAL_COVERAGE:
        failures.append("coverage")
    if failures:
        row["reason"] = "+".join(failures)
        return None
    seed = np.random.SeedSequence([cfg.RANDOM_SEED, key_i[1], key_j[1],
                                  int(key_i[0] == "right"), int(key_j[0] == "right")])
    si, sj = select_evenly_distributed_matches(
        pi, pj, verifier.P, verifier.rows, verifier.cols, np.random.default_rng(seed))
    quality = min(ci, cj) * inlier_ratio * math.log1p(count) / (error + 1e-6)
    edge = ImageEdge(key_i, key_j, "", len(forward), len(ratio), count,
                     inlier_ratio, error, H, pi, pj, si, sj, verifier.P,
                     ci, cj, quality)
    assert_edge_orientation(edge)
    row.update(verified=True, reason="accepted", selected_matches=verifier.P)
    verifier.observations[identity] = edge
    return edge


class IncrementalSIFTRetriever:
    """Replaceable query-before-add descriptor voting interface.

    Each train matrix corresponds to exactly one historical pair. Temporal
    exclusion uses stream positions, not possibly discontinuous frame numbers.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=cfg.FLANN_TREES), dict(checks=cfg.FLANN_CHECKS))
        self.history: list[tuple[int, int]] = []
        self.total_descriptors = 0
        self.last_position = -1

    def describe_pair(self, pair: StereoPair,
                      feature: Callable[[ImageKey], FeatureRecord]) -> np.ndarray:
        matrices = []
        for key in (pair.left_key, pair.right_key):
            record = feature(key)
            indices = np.argsort(-record.responses, kind="stable")[:
                self.cfg.RETRIEVAL_DESCRIPTORS_PER_IMAGE]
            matrices.append(record.descriptors[indices])
        return np.ascontiguousarray(np.vstack(matrices), dtype=np.float32)

    def query(self, descriptors: np.ndarray, position: int) -> list[RetrievalCandidate]:
        assert position > self.last_position, "Retrieval must query only history"
        if self.total_descriptors < 2 or len(descriptors) == 0:
            return []
        votes: Counter = Counter()
        weighted: defaultdict = defaultdict(float)
        for matches in self.matcher.knnMatch(descriptors, k=2):
            if len(matches) != 2:
                continue
            m, n = matches
            if m.distance >= self.cfg.RETRIEVAL_RATIO * n.distance:
                continue
            _, historical_position = self.history[m.imgIdx]
            assert historical_position < position
            if position - historical_position < self.cfg.RETRIEVAL_TEMPORAL_EXCLUSION:
                continue
            votes[m.imgIdx] += 1
            weighted[m.imgIdx] += 1. / (m.distance + 1e-6)
        ranked = sorted(votes, key=lambda idx: (-votes[idx], -weighted[idx], idx))
        return [RetrievalCandidate(self.history[idx][0], self.history[idx][1],
                                   votes[idx], votes[idx] / len(descriptors),
                                   weighted[idx] / len(descriptors))
                for idx in ranked[:self.cfg.RETRIEVAL_TOP_K]]

    def add(self, frame: int, position: int, descriptors: np.ndarray) -> None:
        assert position > self.last_position
        self.last_position = position
        if len(descriptors):
            self.matcher.add([descriptors])
            self.matcher.train()
            self.history.append((frame, position))
            self.total_descriptors += len(descriptors)


def relation_quality(edges: list[ImageEdge], cfg: Config) -> tuple[float, int]:
    best = sorted(edges, key=lambda edge: -edge.quality_score)[:2]
    endpoints = {key for edge in best for key in (edge.key_i, edge.key_j)}
    score = sum(edge.quality_score * (1. if index == 0 else .5)
                for index, edge in enumerate(best))
    return score + cfg.ENDPOINT_COVERAGE_BONUS * len(endpoints), len(endpoints)


def verify_pair_relation(current: StereoPair, historical: StereoPair, source: str,
                         verify: Callable, cfg: Config) -> PairRelation | None:
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


def select_relation_image_edges(relation: PairRelation, role: str,
                                new_pair: StereoPair, historical_nodes: set[ImageKey],
                                intra_edge: ImageEdge | None = None) -> list[ImageEdge]:
    edges = relation.verified_image_edges
    if not edges:
        return []
    if role == "loop":
        ordered = sorted(edges, key=lambda e: -e.quality_score)
        best = ordered[:1]
        endpoints = {best[0].key_i, best[0].key_j}
        for edge in ordered[1:]:
            if (edge.quality_score >= .8 * best[0].quality_score
                    and len(endpoints | {edge.key_i, edge.key_j}) > len(endpoints)):
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
    return [e for e in relation.verified_image_edges
            if e.ransac_inliers >= cfg.LOOP_MIN_RANSAC_INLIERS
            and e.inlier_ratio >= cfg.LOOP_MIN_INLIER_RATIO
            and e.mean_reproj_error <= cfg.LOOP_MAX_REPROJ_ERROR
            and min(e.coverage_i, e.coverage_j) >= cfg.LOOP_MIN_COVERAGE]


def cull_strong_relations(relations: dict[tuple[int, int], PairRelation],
                          cfg: Config) -> int:
    removed = 0
    while True:
        degree: Counter = Counter()
        for relation in relations.values():
            if "strong" in relation.roles:
                degree.update((relation.frame_i, relation.frame_j))
        overloaded = {frame for frame, value in degree.items()
                      if value > cfg.MAX_STRONG_PAIR_DEGREE}
        if not overloaded:
            return removed
        choices = [(key, relation) for key, relation in relations.items()
                   if "strong" in relation.roles
                   and overloaded & {relation.frame_i, relation.frame_j}]
        key, weakest = min(choices, key=lambda item: (item[1].relation_score, item[0]))
        weakest.roles.remove("strong")
        weakest.role_edges.pop("strong", None)
        if not weakest.roles:
            del relations[key]
        removed += 1


def materialize_optimization_graph(intra_edges: list[ImageEdge],
                                    relations: dict[tuple[int, int], PairRelation]
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
        distinct_pairs = {canonical_image_pair(edge.key_i, edge.key_j)
                          for role in relation.roles for edge in relation.role_edges[role]}
        assert len(distinct_pairs) <= 2, "A pair relation may materialize at most two image edges"
        for role in sorted(relation.roles):
            selected = relation.role_edges[role]
            assert 1 <= len(selected) <= 2
            for edge in selected:
                assert any(edge is candidate for candidate in relation.verified_image_edges)
                add(edge, role)
    return result


def find_connected_components(nodes: list[ImageKey], edges: list[ImageEdge]
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


def build_incremental_pair_graph(pairs: list[StereoPair], verifier,
                                  cfg: Config, retriever=None) -> GraphBuildResult:
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
            retrieval_rows.append(dict(query_frame=pair.frame,
                                       query_position=position, **asdict(candidate)))
            counts["retrieval"] += 1
            candidates.setdefault(candidate.frame, "retrieval")
        verified = []
        for frame, source in candidates.items():
            relation = verify_pair_relation(pair, by_frame[frame], source,
                                            verifier.verify, cfg)
            if relation is not None:
                verified.append(relation)
        counts["verified"] += len(verified)
        sequential = [r for r in verified if r.source == "sequential"]
        parent = max(sequential or verified, key=lambda r: r.relation_score, default=None)

        def attach(relation, role, eligible=None):
            identity = tuple(sorted((relation.frame_i, relation.frame_j)))
            stored = relations.setdefault(identity, relation)
            material = relation if eligible is None else replace(
                relation, verified_image_edges=eligible)
            selected = select_relation_image_edges(material, role, pair,
                                                    historical_nodes, intra)
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
        selected_loops = sorted(loops, key=lambda item: -item[0])[:cfg.MAX_LOOP_RELATIONS]
        for _, relation, eligible in selected_loops:
            attach(relation, "loop", eligible)
        loop_ids = {id(r) for _, r, _ in selected_loops}
        strong = [r for r in verified if r is not parent and id(r) not in loop_ids]
        for relation in sorted(strong, key=lambda r: -r.relation_score)[:cfg.MAX_NEW_STRONG_RELATIONS]:
            attach(relation, "strong")
        counts["pruned_strong"] += cull_strong_relations(relations, cfg)
        current_nodes = historical_nodes | {pair.left_key, pair.right_key}
        current_edges = list(materialize_optimization_graph(intra_edges, relations).values())
        components = find_connected_components(list(current_nodes), current_edges)
        reachable = next(group for group in components if pairs[0].left_key in group)
        registered = {pair.left_key, pair.right_key} <= set(reachable)
        if not registered:
            print(f"WARNING: frame {pair.frame}: registration deferred / disconnected pair", flush=True)
        status_rows.append(dict(frame=pair.frame, position=position,
                                 parent_frame=parent.frame_j if parent else None,
                                 registered_at_arrival=registered,
                                 intra_verified=intra is not None,
                                 components_at_arrival=len(components)))
        retriever.add(pair.frame, position, descriptors)
        historical_nodes = current_nodes
        print(f"Pair {position + 1}/{len(pairs)} frame={pair.frame}: "
              f"candidates={len(candidates)}, verified={len(verified)}, "
              f"parent={parent.frame_j if parent else '-'}, "
              f"image_edges={len(current_edges)}, elapsed={time.perf_counter()-started:.1f}s",
              flush=True)
    final = materialize_optimization_graph(intra_edges, relations)
    return GraphBuildResult(relations, intra_edges, final, counts, retrieval_rows, status_rows)


def choose_reference(nodes: list[ImageKey]) -> ImageKey:
    left = [key for key in nodes if key[0] == "left"]
    return min(left or nodes, key=image_order)


def initial_graph_transforms(nodes: list[ImageKey], edges: list[ImageEdge],
                              reference: ImageKey) -> dict[ImageKey, np.ndarray]:
    """Measured-H traversal used ONLY for the pre-affine diagnostic baseline."""
    adjacency: defaultdict = defaultdict(list)
    for edge in edges:
        adjacency[edge.key_i].append((edge, True))
        adjacency[edge.key_j].append((edge, False))
    transforms = {reference: np.eye(3)}
    queue = deque([reference])
    while queue:
        key = queue.popleft()
        for edge, forward in sorted(adjacency[key], key=lambda pair: -pair[0].quality_score):
            other = edge.key_j if forward else edge.key_i
            if other in transforms:
                continue
            # p_j = H_ij p_i, so global H_j = global H_i inv(H_ij).
            step = np.linalg.inv(edge.H_i_to_j) if forward else edge.H_i_to_j
            H = transforms[key] @ step
            H /= H[2, 2]
            if not np.isfinite(H).all():
                raise ValueError("Non-finite initial graph traversal")
            transforms[other] = H
            queue.append(other)
    assert set(transforms) == set(nodes)
    return transforms


def solve_global_affine_sparse(nodes: list[ImageKey], edges: list[ImageEdge],
                               reference: ImageKey, cfg: Config
                               ) -> tuple[dict[ImageKey, np.ndarray], dict]:
    """4DoF shared a=e, b=-d. The fixed identity contribution moves to RHS."""
    started = time.perf_counter()
    variables = [key for key in nodes if key != reference]
    indices = {key: index for index, key in enumerate(variables)}
    if not variables:
        return {reference: np.eye(3)}, dict(stage="affine", status=0,
                                            message="singleton", runtime=0.)
    assert edges
    P = edges[0].selected_matches
    assert all(edge.selected_matches == P for edge in edges)
    residual_count = 2 * len(edges) * P
    rhs = np.zeros(residual_count)
    row_parts, col_parts, value_parts = [], [], []
    for edge_index, edge in enumerate(edges):
        rows = np.arange(edge_index * 2 * P, (edge_index + 1) * 2 * P).reshape(P, 2)
        for key, points, sign in ((edge.key_i, edge.selected_pts_i, 1.),
                                   (edge.key_j, edge.selected_pts_j, -1.)):
            if key == reference:
                rhs[rows.ravel()] -= sign * points.ravel()
                continue
            x, y = points.T
            coefficients = np.zeros((P, 2, 4))
            coefficients[:, 0, 0] = x
            coefficients[:, 0, 1] = y
            coefficients[:, 0, 2] = cfg.TRANSLATION_SCALE
            coefficients[:, 1, 0] = y
            coefficients[:, 1, 1] = -x
            coefficients[:, 1, 3] = cfg.TRANSLATION_SCALE
            row_parts.append(np.repeat(rows.ravel(), 4))
            col_parts.append(np.tile(indices[key] * 4 + np.arange(4), P * 2))
            value_parts.append(sign * coefficients.ravel())
    A = coo_matrix((np.concatenate(value_parts),
                    (np.concatenate(row_parts), np.concatenate(col_parts))),
                   shape=(residual_count, 4 * len(variables))).tocsr()
    A.eliminate_zeros()
    solution = lsmr(A, rhs, atol=1e-11, btol=1e-11,
                    conlim=1e12, maxiter=max(2000, 20 * len(variables)))
    if solution[1] not in (0, 1, 2, 4, 5):
        raise RuntimeError(f"Affine LSMR did not converge: istop={solution[1]}")
    transforms = {reference: np.eye(3)}
    for key, index in indices.items():
        a, b, c, f = solution[0][index * 4:(index + 1) * 4]
        transforms[key] = np.array([[a, b, c * cfg.TRANSLATION_SCALE],
                                     [-b, a, f * cfg.TRANSLATION_SCALE], [0., 0., 1.]])
    assert all(np.isfinite(H).all() for H in transforms.values())
    return transforms, dict(stage="affine", solver="sparse linear LSMR",
                            status=solution[1], iterations=solution[2],
                            residual_norm=solution[3], normal_residual_norm=solution[4],
                            condition_estimate=solution[6], matrix_nnz=A.nnz,
                            runtime=time.perf_counter() - started)


@dataclass
class ProjectiveProblem:
    nodes: list[ImageKey]
    edges: list[ImageEdge]
    reference: ImageKey
    cfg: Config

    def __post_init__(self):
        self.variables = [key for key in self.nodes if key != self.reference]
        self.variable_index = {key: i for i, key in enumerate(self.variables)}
        self.node_index = {key: i for i, key in enumerate(self.nodes)}
        self.variable_node_indices = np.array([self.node_index[key] for key in self.variables], int)
        self.N, self.M = len(self.nodes), len(self.edges)
        self.P = self.edges[0].selected_matches if self.edges else 0
        assert all(e.selected_matches == self.P for e in self.edges)
        self.data_scale = 1. / math.sqrt(self.M * self.P) if self.M else 0.
        self.rigid_scale = math.sqrt(self.cfg.OMEGA_RIGID / self.N)
        self.points = np.stack([
            np.stack([e.selected_pts_i for e in self.edges]),
            np.stack([e.selected_pts_j for e in self.edges])]) if self.M else np.empty((2, 0, 0, 2))
        self.endpoint_node_indices = np.array([
            [self.node_index[e.key_i] for e in self.edges],
            [self.node_index[e.key_j] for e in self.edges]], int)
        self.endpoint_variable_indices = np.array([
            [self.variable_index.get(e.key_i, -1) for e in self.edges],
            [self.variable_index.get(e.key_j, -1) for e in self.edges]], int)
        self.data_rows = 2 * self.M * self.P
        self.residual_count = self.data_rows + 4 * len(self.variables)
        # Build sparsity once; only values change between nonlinear iterations.
        rows, cols = [], []
        for side in range(2):
            valid = self.endpoint_variable_indices[side] >= 0
            row = np.arange(self.data_rows).reshape(self.M, self.P, 2)[valid]
            col = self.endpoint_variable_indices[side, valid, None, None, None] * 8 + np.arange(8)
            rows.append(np.broadcast_to(row[..., None], (*row.shape, 8)).ravel())
            cols.append(np.broadcast_to(col, (*row.shape, 8)).ravel())
        rigid_rows = self.data_rows + np.arange(4 * len(self.variables)).reshape(-1, 4, 1)
        rigid_cols = np.arange(len(self.variables))[:, None, None] * 8 + np.arange(8)
        rows.append(np.broadcast_to(rigid_rows, (len(self.variables), 4, 8)).ravel())
        cols.append(np.broadcast_to(rigid_cols, (len(self.variables), 4, 8)).ravel())
        self.jac_rows, self.jac_cols = np.concatenate(rows), np.concatenate(cols)

    def pack(self, transforms: dict[ImageKey, np.ndarray]) -> np.ndarray:
        packed = np.array([transforms[key].ravel()[:8] for key in self.variables]).reshape(-1, 8)
        packed[:, [2, 5]] /= self.cfg.SIGMA_TR
        return packed.ravel()

    def matrices(self, x: np.ndarray) -> np.ndarray:
        matrices = np.tile(np.eye(3), (self.N, 1, 1))
        packed = x.reshape(-1, 8).copy()
        packed[:, [2, 5]] *= self.cfg.SIGMA_TR
        matrices[self.variable_node_indices, :2, :] = packed[:, :6].reshape(-1, 2, 3)
        matrices[self.variable_node_indices, 2, :2] = packed[:, 6:8]
        return matrices

    def unpack(self, x: np.ndarray) -> dict[ImageKey, np.ndarray]:
        return dict(zip(self.nodes, self.matrices(x)))


def projective_warps(x: np.ndarray, problem: ProjectiveProblem):
    H = problem.matrices(x)
    endpoints = H[problem.endpoint_node_indices]
    numerator = np.einsum("semp,seqp->seqm", endpoints[:, :, :2, :2], problem.points)
    numerator += endpoints[:, :, None, :2, 2]
    raw_w = np.einsum("sep,seqp->seq", endpoints[:, :, 2, :2], problem.points) + 1.
    w = signed_denominator(raw_w)
    return H, numerator / w[..., None], w, raw_w


def rigid_q(matrices: np.ndarray) -> np.ndarray:
    a, b, d, e = (matrices[:, 0, 0], matrices[:, 0, 1],
                   matrices[:, 1, 0], matrices[:, 1, 1])
    g, h = matrices[:, 2, 0], matrices[:, 2, 1]
    return np.stack((a*b + d*e, a*a + d*d - 1.,
                     b*b + e*e - 1., g*g + h*h), axis=1)


def projective_residuals(x: np.ndarray, problem: ProjectiveProblem) -> np.ndarray:
    H, uv, _, _ = projective_warps(x, problem)
    data = (uv[0] - uv[1]).ravel() * problem.data_scale
    # Exactly four Eq.(16) residuals per non-reference image. Identity contributes zero.
    rigid = rigid_q(H[problem.variable_node_indices]).ravel() * problem.rigid_scale
    return np.concatenate((data, rigid))


def projective_sparse_jacobian(x: np.ndarray, problem: ProjectiveProblem) -> csr_matrix:
    H, uv, w, raw_w = projective_warps(x, problem)
    values = []
    for side, sign in ((0, 1.), (1, -1.)):
        valid = problem.endpoint_variable_indices[side] >= 0
        points, denominator, warped = problem.points[side, valid], w[side, valid], uv[side, valid]
        xy_over_w = points / denominator[..., None]
        blocks = np.zeros((int(valid.sum()), problem.P, 2, 8))
        blocks[:, :, 0, :2] = xy_over_w
        blocks[:, :, 0, 2] = problem.cfg.SIGMA_TR / denominator
        blocks[:, :, 1, 3:5] = xy_over_w
        blocks[:, :, 1, 5] = problem.cfg.SIGMA_TR / denominator
        blocks[:, :, :, 6:8] = -warped[..., None] * xy_over_w[:, :, None, :]
        # Inside the epsilon branch w is constant; its derivative is zero.
        blocks[:, :, :, 6:8] *= (np.abs(raw_w[side, valid]) >= 1e-10)[:, :, None, None]
        values.append((blocks * sign * problem.data_scale).ravel())
    variable_H = H[problem.variable_node_indices]
    a, b, d, e = (variable_H[:, 0, 0], variable_H[:, 0, 1],
                   variable_H[:, 1, 0], variable_H[:, 1, 1])
    g, h = variable_H[:, 2, 0], variable_H[:, 2, 1]
    rigid = np.zeros((len(problem.variables), 4, 8))
    rigid[:, 0, 0], rigid[:, 0, 1], rigid[:, 0, 3], rigid[:, 0, 4] = b, a, e, d
    rigid[:, 1, 0], rigid[:, 1, 3] = 2*a, 2*d
    rigid[:, 2, 1], rigid[:, 2, 4] = 2*b, 2*e
    rigid[:, 3, 6], rigid[:, 3, 7] = 2*g, 2*h
    values.append((rigid * problem.rigid_scale).ravel())
    return coo_matrix((np.concatenate(values), (problem.jac_rows, problem.jac_cols)),
                       shape=(problem.residual_count, 8 * len(problem.variables))).tocsr()


def projective_energy(x: np.ndarray, problem: ProjectiveProblem) -> dict[str, float]:
    residual = projective_residuals(x, problem)
    data_energy = float(residual[:problem.data_rows] @ residual[:problem.data_rows])
    raw_rigid = rigid_q(problem.matrices(x))
    rigid_energy = float(np.sum(raw_rigid**2) / problem.N)
    return dict(data_energy=data_energy, rigid_energy=rigid_energy,
                global_energy=data_energy + problem.cfg.OMEGA_RIGID * rigid_energy)


def solve_global_projective(nodes: list[ImageKey], edges: list[ImageEdge],
                            reference: ImageKey, affine: dict[ImageKey, np.ndarray],
                            cfg: Config, verbose: int = 2
                            ) -> tuple[dict[ImageKey, np.ndarray], dict]:
    if len(nodes) == 1:
        return {reference: np.eye(3)}, dict(stage="projective", status=0,
                                            message="singleton", runtime=0.)
    started = time.perf_counter()
    problem = ProjectiveProblem(nodes, edges, reference, cfg)
    x0 = problem.pack(affine)
    before = projective_energy(x0, problem)
    result = least_squares(projective_residuals, x0,
                           jac=projective_sparse_jacobian, args=(problem,),
                           method="trf", tr_solver="lsmr", loss="linear",
                           x_scale="jac", max_nfev=cfg.MAX_NFEV, verbose=verbose)
    after = projective_energy(result.x, problem)
    transforms = problem.unpack(result.x)
    finite = (all(np.isfinite(H).all() for H in transforms.values())
              and np.isfinite(result.fun).all())
    if not finite:
        raise RuntimeError("Projective optimization failed: non-finite result")
    if after["global_energy"] > before["global_energy"] + 1e-8 * max(1., before["global_energy"]):
        raise RuntimeError("Projective optimization failed: objective increased")
    if not result.success:
        print(f"WARNING: projective optimizer did not converge: {result.message}", flush=True)
    summary = dict(stage="projective", solver="TRF/LSMR analytic sparse Jacobian",
                    nfev=result.nfev, njev=result.njev, cost=result.cost,
                    optimality=result.optimality, status=result.status,
                    message=result.message, converged=bool(result.success),
                    finite=finite, runtime=time.perf_counter() - started,
                    M=problem.M, P=problem.P, N=problem.N,
                    omega_rigid=cfg.OMEGA_RIGID, sigma_tr=cfg.SIGMA_TR)
    for name, energy in (("before", before), ("after", after)):
        summary.update({f"{term}_{name}": value for term, value in energy.items()})
    assert np.isclose(2 * result.cost, after["global_energy"], rtol=1e-10)
    return transforms, summary


def edge_errors(edge: ImageEdge, transforms: dict[ImageKey, np.ndarray],
                 selected: bool = False) -> np.ndarray:
    pi = edge.selected_pts_i if selected else edge.inlier_pts_i
    pj = edge.selected_pts_j if selected else edge.inlier_pts_j
    return np.linalg.norm(warp_points(transforms[edge.key_i], pi)
                           - warp_points(transforms[edge.key_j], pj), axis=1)


def error_statistics(errors: np.ndarray) -> dict:
    if not len(errors):
        return dict(count=0, rmse=None, mean=None, median=None, p90=None, p95=None, max=None)
    return dict(count=len(errors), rmse=float(np.sqrt(np.mean(errors**2))),
                mean=float(errors.mean()), median=float(np.median(errors)),
                p90=float(np.percentile(errors, 90)), p95=float(np.percentile(errors, 95)),
                max=float(errors.max()))


def compute_registration_metrics(edges: list[ImageEdge],
                                   stages: dict[str, dict[ImageKey, np.ndarray]]
                                   ) -> tuple[list[dict], list[dict]]:
    rows, summaries = [], []
    for edge in edges:
        row = dict(key_i=key_text(edge.key_i), key_j=key_text(edge.key_j),
                    roles="|".join(sorted(edge.roles)), ransac_inliers=edge.ransac_inliers,
                    selected_matches=edge.selected_matches)
        for stage, transforms in stages.items():
            stat = error_statistics(edge_errors(edge, transforms))
            row[f"rmse_{stage}"] = stat["rmse"]
            if stage == "projective":
                row.update(median_projective=stat["median"], p95_projective=stat["p95"])
        rows.append(row)
    for stage, transforms in stages.items():
        for selected in (False, True):
            arrays = [edge_errors(edge, transforms, selected) for edge in edges]
            errors = np.concatenate(arrays) if arrays else np.array([])
            summaries.append(dict(stage=stage,
                                   points="selected_P" if selected else "all_RANSAC_inliers",
                                   **error_statistics(errors)))
    return rows, summaries


def image_corners(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return np.array([[0., 0.], [width, 0.], [width, height], [0., height]])


def mosaic_bounds(nodes: list[ImageKey], transforms: dict[ImageKey, np.ndarray],
                   shapes: dict[ImageKey, tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    corners = []
    for key in nodes:
        local = image_corners(shapes[key])
        H = transforms[key]
        denominators = local @ H[2, :2] + 1.
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
    width, height = np.maximum(high - low + 2, 1.)
    return min(1., math.sqrt(cfg.MOSAIC_MAX_PIXELS / (width * height)),
               cfg.MOSAIC_MAX_SIDE / width, cfg.MOSAIC_MAX_SIDE / height)


def graphcut_merge(old: np.ndarray, new: np.ndarray, old_mask: np.ndarray,
                    new_mask: np.ndarray, seam_finder, max_pixels: int
                    ) -> tuple[np.ndarray, np.ndarray, bool]:
    if not np.any(old_mask & new_mask):
        choose_new = new_mask > 0
        old[choose_new] = new[choose_new]
        return old, old_mask | new_mask, False
    ratio = min(1., math.sqrt(max_pixels / old_mask.size))
    target = (max(1, round(old.shape[1] * ratio)), max(1, round(old.shape[0] * ratio)))
    resize = target != (old.shape[1], old.shape[0])
    images = [cv2.resize(img, target, interpolation=cv2.INTER_AREA) if resize else img
              for img in (old, new)]
    masks = [cv2.resize(mask, target, interpolation=cv2.INTER_NEAREST) if resize else mask.copy()
             for mask in (old_mask, new_mask)]
    # UMat handles both bindings that mutate masks and bindings that return masks.
    umat_masks = [cv2.UMat(mask) for mask in masks]
    returned = seam_finder.find([cv2.UMat(img.astype(np.float32)) for img in images],
                                [(0, 0), (0, 0)], umat_masks)
    resulting_masks = returned if returned is not None else umat_masks
    seam_new = resulting_masks[1]
    seam_new = seam_new.get() if hasattr(seam_new, "get") else np.asarray(seam_new)
    if resize:
        seam_new = cv2.resize(seam_new, (old.shape[1], old.shape[0]), interpolation=cv2.INTER_NEAREST)
    choose_new = (new_mask > 0) & ((seam_new > 0) | (old_mask == 0))
    old[choose_new] = new[choose_new]
    return old, old_mask | new_mask, True


def render_mosaic(nodes: list[ImageKey], paths: dict[ImageKey, Path],
                   shapes: dict[ImageKey, tuple[int, int]],
                   transforms: dict[ImageKey, np.ndarray], output: Path,
                   cfg: Config, render_scale: float | None = None) -> dict:
    started = time.perf_counter()
    bounds = mosaic_bounds(nodes, transforms, shapes)
    low, high = bounds
    scale = canvas_render_scale(bounds, cfg) if render_scale is None else render_scale
    width, height = np.ceil((high - low) * scale + 2).astype(int)
    if width >= 32767 or height >= 32767:
        raise ValueError("OpenCV canvas size exceeds supported limit")
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas_mask = np.zeros((height, width), np.uint8)
    view = np.array([[scale, 0., -low[0]*scale], [0., scale, -low[1]*scale], [0., 0., 1.]])
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
        shift = np.array([[1., 0., -x0], [0., 1., -y0], [0., 0., 1.]])
        local_H = shift @ render_H
        size = (int(x1-x0), int(y1-y0))
        warped = cv2.warpPerspective(image, local_H, size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full(image.shape[:2], 255, np.uint8), local_H,
                                   size, flags=cv2.INTER_NEAREST)
        old, old_mask = canvas[y0:y1, x0:x1], canvas_mask[y0:y1, x0:x1]
        if seam_finder is not None:
            try:
                merged, union, used = graphcut_merge(old, warped, old_mask, mask,
                                                     seam_finder, cfg.GRAPHCUT_MAX_PIXELS)
                seam_count += int(used)
            except cv2.error as error:
                print(f"WARNING: GraphCut failed for {key_text(key)}: {error}; "
                      "fallback hard overlay for this image", flush=True)
                fallback_count += 1
                merged, union = old, old_mask | mask
                merged[mask > 0] = warped[mask > 0]
        else:
            fallback_count += 1
            merged, union = old, old_mask | mask
            merged[mask > 0] = warped[mask > 0]
        canvas[y0:y1, x0:x1], canvas_mask[y0:y1, x0:x1] = merged, union
        if (index + 1) % 10 == 0 or index + 1 == len(nodes):
            print(f"Mosaic {output.name}: {index+1}/{len(nodes)}, "
                  f"GraphCut seams={seam_count}, elapsed={time.perf_counter()-started:.1f}s", flush=True)
    write_image(output, canvas)
    write_image(output.with_name(output.stem + "_mask.png"), canvas_mask)
    preview_scale = min(1., 1800 / max(width, height))
    preview = cv2.resize(canvas, (round(width*preview_scale), round(height*preview_scale)),
                          interpolation=cv2.INTER_AREA)
    write_image(output.with_name(output.stem + "_preview.jpg"), preview)
    return dict(file=str(output), width=int(width), height=int(height),
                render_scale=scale, input_coordinate_units="original image pixels",
                graphcut_available=seam_finder is not None, graphcut_seams=seam_count,
                hard_overlay_fallbacks=fallback_count,
                graphcut_max_pixels=cfg.GRAPHCUT_MAX_PIXELS,
                valid_pixels=int(np.count_nonzero(canvas_mask)),
                canvas_coverage=float(np.count_nonzero(canvas_mask)/canvas_mask.size),
                view_transform=view.tolist(), runtime=time.perf_counter()-started)


def save_csv(path: Path, rows: list[dict], columns=None) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                default=lambda x: x.item() if isinstance(x, np.generic) else str(x)),
                    encoding="utf-8")


def graph_statistics(pairs, nodes, graph: GraphBuildResult) -> dict:
    edges = list(graph.optimization_edges_by_pair.values())
    degree = Counter({key: 0 for key in nodes})
    for edge in edges:
        degree.update((edge.key_i, edge.key_j))
    values = np.array(list(degree.values()))
    stats = dict(num_stereo_pairs=len(pairs), num_images=len(nodes),
                  intra_pair_edges=sum("intra_pair" in e.roles for e in edges),
                  total_optimization_edges=len(edges),
                  image_graph_connected_components=len(find_connected_components(nodes, edges)),
                  degree_min=int(values.min()), degree_mean=float(values.mean()),
                  degree_median=float(np.median(values)), degree_max=int(values.max()))
    for role in ("tree", "strong", "loop"):
        stats[f"pair_{role}_relations"] = sum(role in r.roles for r in graph.relations.values())
        stats[f"image_{role}_edges"] = sum(role in e.roles for e in edges)
    print("\n" + "="*60 + "\nINCREMENTAL GRAPH CONSTRUCTION\n" + "="*60)
    print(f"Stereo pairs = {len(pairs)}\nImages = {len(nodes)}")
    print(f"Sequential pair candidates = {graph.candidate_counts['sequential']}")
    print(f"Retrieval pair candidates = {graph.candidate_counts['retrieval']}")
    print(f"Verified pair relations = {graph.candidate_counts['verified']}")
    for role in ("tree", "strong", "loop"):
        print(f"{role.title()} relations = {stats[f'pair_{role}_relations']}")
    print("\n" + "="*60 + "\nIMAGE OPTIMIZATION GRAPH\n" + "="*60)
    print(f"Nodes = {len(nodes)}\nIntra-pair ordinary edges = {stats['intra_pair_edges']}")
    for role in ("tree", "strong", "loop"):
        print(f"{role.title()} ordinary edges = {stats[f'image_{role}_edges']}")
    print(f"Total ordinary edges = {len(edges)}\nConnected components = "
          f"{stats['image_graph_connected_components']}")
    print(f"Degree min/median/mean/max = {values.min()}/{np.median(values)}/"
          f"{values.mean():.3f}/{values.max()}")
    print("IMPORTANT:\nAll optimization edges are ordinary = True\nStereo-special edges = 0")
    if values.mean() > 10:
        print("WARNING: optimization graph is unexpectedly dense")
    return stats


def save_graph_diagnostics(output: Path, pairs, paths, graph: GraphBuildResult,
                            verifier: VisualVerifier, cfg: Config) -> dict:
    data = output / "data"
    pair_columns = ["frame_i", "frame_j", "roles", "relation_score", "endpoint_coverage",
                    "num_verified_LL_LR_RL_RR", "source"]
    pair_rows = [dict(frame_i=r.frame_i, frame_j=r.frame_j, roles="|".join(sorted(r.roles)),
                       relation_score=r.relation_score, endpoint_coverage=r.endpoint_coverage,
                       num_verified_LL_LR_RL_RR=len(r.verified_image_edges), source=r.source)
                 for r in graph.relations.values()]
    save_csv(data / "pair_topology_graph.csv", pair_rows, pair_columns)
    edge_rows = []
    arrays = {}
    for index, edge in enumerate(graph.optimization_edges_by_pair.values()):
        row = {name: getattr(edge, name) for name in
               ("raw_matches", "ratio_matches", "ransac_inliers", "inlier_ratio",
                "mean_reproj_error", "coverage_i", "coverage_j", "quality_score", "selected_matches")}
        row.update(key_i=key_text(edge.key_i), key_j=key_text(edge.key_j),
                    roles="|".join(sorted(edge.roles)), provenance=edge.provenance,
                    observation_index=index)
        edge_rows.append(row)
        for name in ("H_i_to_j", "inlier_pts_i", "inlier_pts_j", "selected_pts_i", "selected_pts_j"):
            arrays[f"e{index}_{name}"] = getattr(edge, name)
    edge_columns = ["key_i", "key_j", "roles", "provenance", "raw_matches", "ratio_matches",
                    "ransac_inliers", "inlier_ratio", "mean_reproj_error", "coverage_i",
                    "coverage_j", "quality_score", "selected_matches", "observation_index"]
    save_csv(data / "image_optimization_graph.csv", edge_rows, edge_columns)
    np.savez_compressed(data / "optimization_observations.npz", **arrays)
    save_csv(data / "matching_diagnostics.csv", verifier.diagnostics)
    save_csv(data / "retrieval_candidates.csv", graph.retrieval_rows,
              ["query_frame", "query_position", "frame", "position", "votes", "score", "weighted_score"])
    save_csv(data / "pair_registration_status.csv", graph.pair_status_rows)
    manifest = []
    for key, path in paths.items():
        feature = verifier.feature_cache[key]
        manifest.append(dict(camera=key[0], frame=key[1], path=str(path.resolve()),
                               bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                               height=feature.shape[0], width=feature.shape[1],
                               sift_features=len(feature.descriptors),
                               sift_extractions=verifier.extraction_counts[key]))
    save_csv(data / "input_manifest.csv", manifest)
    stats = graph_statistics(pairs, list(paths), graph)
    stats.update(graph.candidate_counts)
    save_json(data / "graph_summary.json", stats)
    save_json(data / "graph_config.json", asdict(cfg))
    return stats


def load_saved_graph(output: Path, paths: dict[ImageKey, Path], cfg: Config):
    """Explicit replay of this implementation's final graph, with input validation."""
    data = output / "data"
    recorded_cfg = json.loads((data / "graph_config.json").read_text(encoding="utf-8"))
    current_cfg = json.loads(json.dumps(asdict(cfg)))
    rendering_options = {"MOSAIC_MAX_PIXELS", "MOSAIC_MAX_SIDE", "GRAPHCUT_MAX_PIXELS"}
    if ({k: v for k, v in recorded_cfg.items() if k not in rendering_options}
            != {k: v for k, v in current_cfg.items() if k not in rendering_options}):
        raise ValueError("Saved graph config differs; use a new output directory")
    manifest = pd.read_csv(data / "input_manifest.csv")
    shapes = {}
    if len(manifest) != len(paths):
        raise ValueError("Saved graph image count differs")
    for row in manifest.itertuples(index=False):
        key = (row.camera, int(row.frame))
        path = paths.get(key)
        if (path is None or str(path.resolve()) != row.path
                or path.stat().st_size != row.bytes or path.stat().st_mtime_ns != row.mtime_ns):
            raise ValueError(f"Input changed since graph construction: {key}")
        shapes[key] = (int(row.height), int(row.width))
    edges = []
    with np.load(data / "optimization_observations.npz", allow_pickle=False) as arrays:
        for row in pd.read_csv(data / "image_optimization_graph.csv").to_dict("records"):
            index = row.pop("observation_index")
            for name in ("key_i", "key_j"):
                camera, frame = row[name].split(":")
                row[name] = (camera, int(frame))
            row["roles"] = set(row["roles"].split("|"))
            for name in ("H_i_to_j", "inlier_pts_i", "inlier_pts_j", "selected_pts_i", "selected_pts_j"):
                row[name] = arrays[f"e{index}_{name}"].copy()
            edge = ImageEdge(**row)
            assert_edge_orientation(edge)
            edges.append(edge)
    return edges, shapes, json.loads((data / "graph_summary.json").read_text(encoding="utf-8"))


def save_transforms(data: Path, nodes, initial, affine, projective,
                     component_ids, references) -> None:
    affine_rows, projective_rows, initial_rows = [], [], []
    names = ("a", "b", "c", "d", "e", "f", "g", "h")
    for key in sorted(nodes, key=image_order):
        common = dict(camera=key[0], frame=key[1], component=component_ids[key],
                        is_reference=key == references[component_ids[key]])
        affine_rows.append(dict(common, **dict(zip(names[:6], affine[key].ravel()[:6]))))
        row = dict(common, delta_norm=float(np.linalg.norm(projective[key] - affine[key])))
        row.update({f"affine_init_{name}": value for name, value in zip(names, affine[key].ravel()[:8])})
        row.update({f"projective_final_{name}": value for name, value in zip(names, projective[key].ravel()[:8])})
        projective_rows.append(row)
        initial_rows.append(dict(common, **dict(zip(names, initial[key].ravel()[:8]))))
    save_csv(data / "global_affine_transforms.csv", affine_rows)
    save_csv(data / "global_projective_transforms.csv", projective_rows)
    save_csv(data / "initial_traversal_transforms.csv", initial_rows)


def run_registration_and_mosaics(paths, edges, shapes, output: Path, cfg: Config):
    nodes, data = list(paths), output / "data"
    components = find_connected_components(nodes, edges)
    initial, affine, projective = {}, {}, {}
    component_ids, references, optimizer_rows = {}, {}, []
    for component_id, group in enumerate(components):
        members = set(group)
        component_edges = [edge for edge in edges if edge.key_i in members]
        assert all(edge.key_j in members for edge in component_edges)
        reference = choose_reference(group)
        references[component_id] = reference
        component_ids.update({key: component_id for key in group})
        initial.update(initial_graph_transforms(group, component_edges, reference))
        print("\n" + "="*60 + "\nGLOBAL AFFINE OPTIMIZATION\n" + "="*60, flush=True)
        print(f"Component={component_id}, N={len(group)}, reference={key_text(reference)}\n"
              f"M = {len(component_edges)}\nP = {point_policy(len(nodes))[0]}\n"
              "solver = sparse linear LSMR", flush=True)
        affine_result, affine_info = solve_global_affine_sparse(group, component_edges, reference, cfg)
        affine.update(affine_result)
        _, affine_metrics = compute_registration_metrics(component_edges,
                                                          dict(initial=initial, affine=affine))
        print(f"RMSE before = {affine_metrics[0]['rmse']}\n"
              f"RMSE after = {affine_metrics[2]['rmse']}", flush=True)
        print("\n" + "="*60 + "\nGLOBAL PROJECTIVE OPTIMIZATION\n" + "="*60, flush=True)
        print("Objective: Eproj = E_match/(M*P) + 800 * E_rigid/N\n"
              "sigma_tr = 5000\nNo GPS prior\nNo stereo residual\n"
              "No safety penalty\nNo robust loss", flush=True)
        result, projective_info = solve_global_projective(group, component_edges, reference,
                                                           affine_result, cfg)
        projective.update(result)
        projective_errors = [edge_errors(edge, result) for edge in component_edges]
        projective_rmse = error_statistics(np.concatenate(projective_errors) if projective_errors
                                           else np.array([]))["rmse"]
        print(f"RMSE before = {affine_metrics[2]['rmse']}\nRMSE after = {projective_rmse}", flush=True)
        for info in (affine_info, projective_info):
            optimizer_rows.append(dict(component=component_id, **info))
        print(json.dumps(projective_info, indent=2, default=str), flush=True)
        save_csv(data / "optimizer_summary.csv", optimizer_rows)
    save_transforms(data, nodes, initial, affine, projective, component_ids, references)
    edge_metrics, summary_metrics = compute_registration_metrics(
        edges, dict(initial=initial, affine=affine, projective=projective))
    save_csv(data / "edge_registration_metrics.csv", edge_metrics)
    save_csv(data / "registration_metrics_summary.csv", summary_metrics)
    print("\nAll-RANSAC paper-style and selected-P metrics:\n"
          + pd.DataFrame(summary_metrics).to_string(index=False), flush=True)
    mosaic_rows = []
    for component_id, group in enumerate(components):
        scale = min(canvas_render_scale(mosaic_bounds(group, stage, shapes), cfg)
                    for stage in (affine, projective))
        for stage_name, transforms in (("affine", affine), ("projective", projective)):
            filename = f"global_{stage_name}_mosaic"
            if len(components) > 1:
                filename += f"_component_{component_id:03d}"
            path = output / "mosaics" / f"{filename}.jpg"
            info = render_mosaic(group, paths, shapes, transforms, path, cfg, scale)
            mosaic_rows.append(dict(stage=stage_name, component=component_id, **info))
            if len(components) > 1 and component_id == 0:
                for suffix in (".jpg", "_preview.jpg", "_mask.png"):
                    shutil.copy2(path.with_name(filename + suffix),
                                  path.with_name(f"global_{stage_name}_mosaic" + suffix))
            save_json(data / "mosaic_summary.json", mosaic_rows)
    return summary_metrics, optimizer_rows, mosaic_rows


def synthetic_edge(ki, kj, Hi=None, Hj=None, count=80, quality=1., seed=7):
    Hi = np.eye(3) if Hi is None else Hi
    Hj = np.eye(3) if Hj is None else Hj
    rng = np.random.default_rng(seed)
    pi = rng.uniform([30., 20.], [900., 700.], (count, 2))
    H = np.linalg.inv(Hj) @ Hi
    H /= H[2, 2]
    pj = warp_points(H, pi)
    si, sj = select_evenly_distributed_matches(pi, pj, 40, 5, 8, rng)
    error = float(np.linalg.norm(warp_points(H, pi)-pj, axis=1).mean())
    return ImageEdge(ki, kj, "tree", count, count, count, 1., error,
                     H, pi, pj, si, sj, 40, .4, .4, quality, {"tree"})


def synthetic_problem(count=7, projective=True):
    nodes = [("left", index) for index in range(count)]
    transforms = {}
    for i, key in enumerate(nodes):
        theta = .015 * i
        c, s = np.cos(theta), np.sin(theta)
        transforms[key] = np.array([[c, s, 80.*i], [-s, c, 12.*i],
                                      [1.5e-5*i if projective else 0.,
                                       -8e-6*i if projective else 0., 1.]])
    edges = [synthetic_edge(nodes[i], nodes[j], transforms[nodes[i]], transforms[nodes[j]],
                             seed=100*i+j) for i in range(count) for j in range(i+1, count)
             if j-i <= 2 or (i == 0 and j == count-1)]
    return nodes, edges, transforms


class SelfTests(unittest.TestCase):
    def test_ordinary_edge_orientation(self):
        Hi = np.array([[1., .02, 30.], [-.01, .98, 17.], [1e-5, 0., 1.]])
        edge = synthetic_edge(("right", 10), ("left", 2), Hi)
        assert_edge_orientation(edge)
        identity = canonical_image_pair(edge.key_i, edge.key_j)
        self.assertEqual(edge.key_i, ("right", 10))
        self.assertEqual(identity[0], ("left", 2))
        reversed_edge = reverse_edge(edge)
        self.assertEqual(reversed_edge.key_i, edge.key_j)
        np.testing.assert_allclose(reversed_edge.inlier_pts_i, edge.inlier_pts_j)
        np.testing.assert_allclose(warp_points(reversed_edge.H_i_to_j,
                                               reversed_edge.inlier_pts_i),
                                    reversed_edge.inlier_pts_j, atol=1e-8)
        self.assertEqual(reversed_edge.coverage_i, edge.coverage_j)
        np.testing.assert_allclose(reverse_edge(reversed_edge).H_i_to_j, edge.H_i_to_j, atol=1e-12)

    def test_uniform_P_selection(self):
        rng = np.random.default_rng(7)
        points = rng.uniform(0, 100, (400, 2))
        for total in (140, 301):
            P, rows, cols = point_policy(total)
            si, sj = select_evenly_distributed_matches(points, points+1, P, rows, cols,
                                                        np.random.default_rng(7))
            self.assertEqual(si.shape, (P, 2))
            self.assertEqual(len(np.unique(si, axis=0)), P)
            np.testing.assert_allclose(sj, si+1)
            repeat, _ = select_evenly_distributed_matches(points, points+1, P, rows, cols,
                                                           np.random.default_rng(7))
            np.testing.assert_array_equal(si, repeat)
        with self.assertRaises(ValueError):
            select_evenly_distributed_matches(points[:10], points[:10], 40, 5, 8, rng)

    def test_candidate_is_not_optimization_edge(self):
        _candidate = RetrievalCandidate(frame=0, position=0, votes=900, score=.9, weighted_score=20.)
        relation = PairRelation(10, 0, "retrieval", [], 100., 0)
        self.assertEqual(materialize_optimization_graph([], {(0, 10): relation}), {})
        calls = []
        def reject(ki, kj):
            calls.append((ki, kj))
            return None
        a, b = StereoPair(10, ("left", 10), ("right", 10)), StereoPair(0, ("left", 0), ("right", 0))
        self.assertIsNone(verify_pair_relation(a, b, "retrieval", reject, Config()))
        self.assertEqual(len(calls), 4)

    def test_all_edges_are_ordinary(self):
        import ast
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        edge_fields = set(ImageEdge.__dataclass_fields__)
        self.assertNotIn("edge_type", edge_fields)
        for item in ast.walk(tree):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.assertFalse("stereo" in item.name and "residual" in item.name)
        self.assertEqual(set(ProjectiveProblem.__dataclass_fields__),
                         {"nodes", "edges", "reference", "cfg"})

    def test_pair_relation_materialization(self):
        new = StereoPair(1, ("left", 1), ("right", 1))
        old = StereoPair(0, ("left", 0), ("right", 0))
        edges = [synthetic_edge(ki, kj, quality=10. if ki[0] != kj[0] else 1.)
                 for ki in (new.left_key, new.right_key) for kj in (old.left_key, old.right_key)]
        relation = PairRelation(1, 0, "sequential", edges, 10., 4)
        selected = select_relation_image_edges(relation, "tree", new, {old.left_key, old.right_key})
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({key for e in selected for key in (e.key_i, e.key_j)}), 4)
        self.assertTrue(all(e.key_i[0] != e.key_j[0] for e in selected))
        one = replace(relation, verified_image_edges=edges[:1])
        intra = synthetic_edge(new.left_key, new.right_key)
        selected = select_relation_image_edges(one, "tree", new, {old.left_key, old.right_key}, intra)
        components = find_connected_components([new.left_key, new.right_key, old.left_key], selected+[intra])
        self.assertEqual(len(components), 1)

    def test_tree_connectivity(self):
        import contextlib
        import io
        pairs = [StereoPair(i, ("left", i), ("right", i)) for i in range(6)]
        class FakeVerifier:
            def verify(self, ki, kj):
                return synthetic_edge(ki, kj)
            def feature(self, key):
                return None
        class FakeRetriever:
            def describe_pair(self, pair, feature):
                return np.empty((0, 128), np.float32)
            def query(self, descriptors, position):
                return []
            def add(self, frame, position, descriptors):
                pass
        with contextlib.redirect_stdout(io.StringIO()):
            graph = build_incremental_pair_graph(pairs, FakeVerifier(), Config(), FakeRetriever())
        self.assertTrue(all(row["registered_at_arrival"] for row in graph.pair_status_rows))
        self.assertEqual(sum("tree" in r.roles for r in graph.relations.values()), 5)
        nodes = [key for pair in pairs for key in (pair.left_key, pair.right_key)]
        tree_only = [e for e in graph.optimization_edges_by_pair.values()
                     if "tree" in e.roles or "intra_pair" in e.roles]
        self.assertEqual(len(find_connected_components(nodes, tree_only)), 1)
        # A missing cross link must be reported, never fabricated.
        class BrokenVerifier(FakeVerifier):
            def verify(self, ki, kj):
                return synthetic_edge(ki, kj) if ki[1] == kj[1] else None
        with contextlib.redirect_stdout(io.StringIO()):
            broken = build_incremental_pair_graph(pairs, BrokenVerifier(), Config(), FakeRetriever())
        self.assertFalse(broken.pair_status_rows[-1]["registered_at_arrival"])
        self.assertEqual(len(find_connected_components(nodes, list(broken.optimization_edges_by_pair.values()))), 6)

    def test_strong_edge_culling(self):
        relations = {}
        for i in range(1, 7):
            edge = synthetic_edge(("left", 0), ("left", i))
            relations[(0, i)] = PairRelation(0, i, "sequential", [edge], float(i), 2,
                                              {"strong"}, {"strong": [edge]})
        relations[(0, 1)].roles.update(("tree", "loop"))
        relations[(0, 1)].role_edges.update(tree=relations[(0, 1)].verified_image_edges,
                                            loop=relations[(0, 1)].verified_image_edges)
        self.assertEqual(cull_strong_relations(relations, Config()), 2)
        self.assertEqual(relations[(0, 1)].roles, {"tree", "loop"})
        self.assertNotIn((0, 2), relations)
        self.assertEqual(sum("strong" in r.roles for r in relations.values()), 4)
        final = materialize_optimization_graph([], relations)
        self.assertNotIn(canonical_image_pair(("left", 0), ("left", 2)), final)

    def test_affine_sparse_recovery(self):
        nodes, edges, truth = synthetic_problem(projective=False)
        result, info = solve_global_affine_sparse(nodes, edges, nodes[0], Config())
        errors = np.concatenate([edge_errors(edge, result) for edge in edges])
        self.assertLess(float(np.sqrt(np.mean(errors**2))), 1e-5)
        for key in nodes:
            np.testing.assert_allclose(result[key], truth[key], atol=1e-4)
        self.assertIn(info["status"], (1, 2, 4, 5))

    def test_projective_objective_exact_terms(self):
        nodes, edges, truth = synthetic_problem(count=3)
        problem = ProjectiveProblem(nodes, edges, nodes[0], Config())
        x = problem.pack(truth)
        x[0] += .03
        x[1] += .01
        transforms = problem.unpack(x)
        residual = projective_residuals(x, problem)
        self.assertEqual(len(residual), 2*len(edges)*40 + 4*(len(nodes)-1))
        matching = sum(np.sum(edge_errors(edge, transforms, True)**2) for edge in edges)/(len(edges)*40)
        rigid = 0.
        for H in transforms.values():
            a, b, c, d, e, f, g, h = H.ravel()[:8]
            rigid += (a*b+d*e)**2 + (a*a+d*d-1)**2 + (b*b+e*e-1)**2 + (g*g+h*h)**2
        manual = matching + 800*rigid/len(nodes)
        self.assertAlmostEqual(float(residual @ residual), manual, places=8)
        np.testing.assert_allclose(projective_energy(x, problem)["global_energy"], manual)
        # Arbitrary external settings cannot affect the mathematical context.
        unused_settings = {"unused_parameter": 1.}
        unused_settings["unused_parameter"] = 1e12
        np.testing.assert_array_equal(residual, projective_residuals(x, problem))

    def test_projective_analytic_jacobian(self):
        nodes, edges, truth = synthetic_problem(count=3)
        problem = ProjectiveProblem(nodes, edges, nodes[0], Config())
        x = problem.pack(truth)
        x[0] += .01
        analytic = projective_sparse_jacobian(x, problem).toarray()
        numerical = np.zeros_like(analytic)
        for column in range(len(x)):
            step = 1e-9 if column % 8 >= 6 else 1e-6
            offset = np.zeros_like(x)
            offset[column] = step
            numerical[:, column] = (projective_residuals(x+offset, problem)
                                      - projective_residuals(x-offset, problem))/(2*step)
        relative = float(np.linalg.norm(analytic-numerical)/np.linalg.norm(numerical))
        column_relative = np.linalg.norm(analytic-numerical, axis=0)/np.maximum(np.linalg.norm(numerical, axis=0), 1.)
        print(f"Jacobian relative error={relative:.3e}; max column error={column_relative.max():.3e}")
        self.assertLess(relative, 1e-5)
        self.assertLess(float(column_relative.max()), 1e-5)

    def test_projective_recovery(self):
        nodes, edges, _ = synthetic_problem(count=7)
        affine, _ = solve_global_affine_sparse(nodes, edges, nodes[0], Config())
        result, info = solve_global_projective(nodes, edges, nodes[0], affine, Config(), verbose=0)
        before = np.concatenate([edge_errors(e, affine) for e in edges])
        after = np.concatenate([edge_errors(e, result) for e in edges])
        self.assertLess(float(np.mean(after**2)), float(np.mean(before**2)))
        self.assertLess(info["global_energy_after"], info["global_energy_before"])
        self.assertTrue(all(np.isfinite(H).all() for H in result.values()))
        np.testing.assert_array_equal(result[nodes[0]], np.eye(3))
        print(f"Synthetic recovery all-inlier RMSE: {np.sqrt(np.mean(before**2)):.6f} -> "
              f"{np.sqrt(np.mean(after**2)):.6f}")

    def test_mutual_ratio_matching(self):
        def match(q, t, distance):
            return cv2.DMatch(q, t, distance)
        forward = [[match(0, 1, 1.), match(0, 0, 4.)],
                   [match(1, 0, 1.), match(1, 1, 4.)]]
        backward = [[match(0, 0, 1.), match(0, 1, 4.)],
                    [match(1, 0, 1.), match(1, 1, 4.)]]
        ratio, mutual = mutual_ratio_matches(forward, backward, .75)
        self.assertEqual(len(ratio), 2)
        self.assertEqual(mutual, [(0, 1)])

    def test_retrieval_is_incremental(self):
        cfg = replace(Config(), RETRIEVAL_TEMPORAL_EXCLUSION=2)
        retriever = IncrementalSIFTRetriever(cfg)
        descriptors = np.random.default_rng(7).uniform(0, 100, (100, 128)).astype(np.float32)
        self.assertEqual(retriever.query(descriptors, 0), [])
        retriever.add(100, 0, descriptors)
        self.assertEqual(retriever.query(descriptors, 1), [])
        candidates = retriever.query(descriptors, 2)
        self.assertEqual(candidates[0].frame, 100)
        self.assertEqual(candidates[0].votes, 100)
        with self.assertRaises(AssertionError):
            retriever.query(descriptors, 0)

    def test_loop_strict_gate_and_second_edge(self):
        a = synthetic_edge(("left", 20), ("left", 0), quality=10.)
        b = synthetic_edge(("right", 20), ("right", 0), quality=7.9)
        relation = PairRelation(20, 0, "retrieval", [a, b], 20., 4)
        pair = StereoPair(20, ("left", 20), ("right", 20))
        selected = select_relation_image_edges(relation, "loop", pair, set())
        self.assertEqual(len(selected), 1)
        b.quality_score = 8.1
        self.assertEqual(len(select_relation_image_edges(relation, "loop", pair, set())), 2)
        b.inlier_ratio = .3
        strict = strict_loop_edges(relation, Config())
        self.assertEqual(len(strict), 1)
        self.assertIs(strict[0], a)

    def test_dedup_roles_and_orientation(self):
        edge = synthetic_edge(("right", 1), ("left", 0))
        reverse = reverse_edge(edge)
        relation = PairRelation(1, 0, "sequential", [edge, reverse], 1., 2,
                                  {"tree", "strong"}, {"tree": [edge], "strong": [reverse]})
        graph = materialize_optimization_graph([], {(0, 1): relation})
        self.assertEqual(len(graph), 1)
        result = next(iter(graph.values()))
        self.assertEqual(result.roles, {"tree", "strong"})
        assert_edge_orientation(result)

    def test_graphcut_mask_union(self):
        if not hasattr(cv2, "detail_GraphCutSeamFinder"):
            self.skipTest("GraphCut unavailable")
        old = np.full((60, 100, 3), 50, np.uint8)
        new = np.full_like(old, 150)
        old_mask, new_mask = np.zeros((60, 100), np.uint8), np.zeros((60, 100), np.uint8)
        old_mask[:, :70] = 255
        new_mask[:, 30:] = 255
        merged, union, used = graphcut_merge(old, new, old_mask, new_mask,
                                             cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD"), 2000)
        self.assertTrue(used)
        self.assertTrue(np.all(union == 255))
        self.assertTrue(np.all(merged[:, :20] == 50))
        self.assertTrue(np.all(merged[:, 80:] == 150))

    def test_retrieval_parent_materializes_at_most_two_edges(self):
        import contextlib
        import io
        pairs = [StereoPair(i, ("left", i), ("right", i)) for i in (0, 10)]
        class FakeVerifier:
            def verify(self, ki, kj):
                quality = {("left", "left"): 10., ("left", "right"): 9.,
                           ("right", "left"): .1, ("right", "right"): 1.}[ki[0], kj[0]]
                return synthetic_edge(ki, kj, quality=quality)
            def feature(self, key):
                return None
        class FakeRetriever:
            def describe_pair(self, pair, feature):
                return np.empty((0, 128), np.float32)
            def query(self, descriptors, position):
                return [RetrievalCandidate(0, 0, 100, 1., 1.)] if position else []
            def add(self, frame, position, descriptors):
                pass
        cfg = replace(Config(), SEQUENTIAL_OFFSETS=(2, 4), RETRIEVAL_TEMPORAL_EXCLUSION=1)
        with contextlib.redirect_stdout(io.StringIO()):
            graph = build_incremental_pair_graph(pairs, FakeVerifier(), cfg, FakeRetriever())
        relation = graph.relations[(0, 10)]
        self.assertEqual(relation.roles, {"tree"})
        self.assertEqual(len(relation.role_edges["tree"]), 2)
        cross = [edge for edge in graph.optimization_edges_by_pair.values()
                 if edge.key_i[1] != edge.key_j[1]]
        self.assertEqual(len(cross), 2)
        self.assertTrue(graph.pair_status_rows[-1]["registered_at_arrival"])

    def test_existing_run_is_not_overwritten(self):
        import contextlib
        import io
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(prefix="uav_graph_test_") as directory:
            output = Path(directory)
            (output / "data").mkdir()
            (output / "data" / "image_optimization_graph.csv").write_text("existing graph")
            (output / "run.log").write_text("original successful log")
            args = argparse.Namespace(output=output, self_test=False, resume_graph=False,
                                      opencv_threads=1, left=LEFT_IN, right=RIGHT_IN)
            with patch(__name__ + ".parse_arguments", return_value=args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 1)
            self.assertEqual((output / "run.log").read_text(), "original successful log")
            self.assertFalse((output / "run_failure.json").exists())


def run_self_tests(output: Path | None = None) -> bool:
    cv2.setRNGSeed(7)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelfTests)
    names = [test.id().split(".")[-1] for test in suite]
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    failed = {test.id().split(".")[-1] for test, _ in result.failures + result.errors}
    skipped = {test.id().split(".")[-1] for test, _ in result.skipped}
    report = dict(tests_run=result.testsRun, failures=len(result.failures),
                    errors=len(result.errors), skipped=len(result.skipped),
                    success=result.wasSuccessful(), runtime=time.perf_counter()-started,
                    tests={name: "FAIL" if name in failed else "SKIP" if name in skipped else "PASS"
                           for name in names})
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        save_json(output / "self_test_results.json", report)
    print("SELF_TESTS_PASS" if result.wasSuccessful() else "SELF_TESTS_FAIL", flush=True)
    return result.wasSuccessful()


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile
    def write(self, text):
        self.terminal.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return len(text)
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, default=LEFT_IN)
    parser.add_argument("--right", type=Path, default=RIGHT_IN)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--resume-graph", action="store_true",
                        help="Replay this program's final saved graph after input/config validation")
    parser.add_argument("--opencv-threads", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args, cfg = parse_arguments(), Config()
    if (not args.self_test and not args.resume_graph
            and (args.output / "data" / "image_optimization_graph.csv").exists()):
        print("ERROR: Results already exist. Use --resume-graph or a new --output directory.")
        return 1
    cv2.setNumThreads(args.opencv_threads)
    cv2.setRNGSeed(cfg.RANDOM_SEED)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        with (args.output / "self_test.log").open("w", encoding="utf-8") as logfile:
            previous = sys.stdout
            sys.stdout = Tee(previous, logfile)
            try:
                return 0 if run_self_tests(args.output) else 1
            finally:
                sys.stdout = previous
    for name in ("data", "mosaics"):
        (args.output / name).mkdir(exist_ok=True)
    logname = "run_resume.log" if args.resume_graph else "run.log"
    with (args.output / logname).open("w", encoding="utf-8") as logfile:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, logfile), Tee(stderr, logfile)
        started = time.perf_counter()
        try:
            print(f"Python = {sys.executable}\nOpenCV = {cv2.__version__}\n"
                  f"Output = {args.output.resolve()}", flush=True)
            pairs, paths = load_stereo_pairs(args.left, args.right, cfg)
            print(f"Synchronized pairs = {len(pairs)}, images = {len(paths)}, "
                  f"P/grid = {point_policy(len(paths))}", flush=True)
            if args.resume_graph:
                edges, shapes, stats = load_saved_graph(args.output, paths, cfg)
                print("Replaying validated FINAL image optimization graph", flush=True)
            else:
                verifier = VisualVerifier(paths, len(paths), cfg)
                graph = build_incremental_pair_graph(pairs, verifier, cfg)
                stats = save_graph_diagnostics(args.output, pairs, paths, graph, verifier, cfg)
                edges = list(graph.optimization_edges_by_pair.values())
                shapes = {key: record.shape for key, record in verifier.feature_cache.items()}
                assert set(shapes) == set(paths)
                assert all(count == 1 for count in verifier.extraction_counts.values())
                del verifier, graph
            metrics, optimizer, mosaics = run_registration_and_mosaics(
                paths, edges, shapes, args.output, cfg)
            summary = dict(status="completed", graph=stats, metrics=metrics,
                             optimizer=optimizer, mosaics=mosaics,
                             runtime=time.perf_counter()-started)
            save_json(args.output / "run_summary.json", summary)
            print(f"RUN_COMPLETED in {summary['runtime']:.2f} seconds", flush=True)
            return 0
        except Exception:
            traceback.print_exc()
            save_json(args.output / "run_failure.json",
                       dict(status="failed", traceback=traceback.format_exc(),
                            runtime=time.perf_counter()-started))
            return 1
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    raise SystemExit(main())
