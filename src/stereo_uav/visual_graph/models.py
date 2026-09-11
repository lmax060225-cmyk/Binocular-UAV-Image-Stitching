"""Models for the visual graph algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ImageKey = tuple[str, int]


ImagePair = tuple[ImageKey, ImageKey]


BASE_DIR = Path(__file__).resolve().parents[3]


LEFT_IN = BASE_DIR / "data_set_2" / "left"


RIGHT_IN = BASE_DIR / "data_set_2" / "right"


OUTPUT_DIR = BASE_DIR / "out_new_visual_graph_test"


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
