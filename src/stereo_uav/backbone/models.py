"""Models for the backbone algorithm."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ImageRecord:
    """Input image and basic metadata."""

    index: int  # Image record index.
    path: Path  # Input image path.
    name: str  # Input filename.
    image: Optional[np.ndarray]  # Original BGR pixels, released after leaving the active window.
    gray: Optional[
        np.ndarray
    ]  # Grayscale pixels, released after feature extraction and leaving the active window.
    width: int  # Width in pixels.
    height: int  # Height in pixels.


@dataclass
class FeatureRecord:
    """SIFT features for one image."""

    keypoints: List[cv2.KeyPoint]  # Keypoints for this image.
    # Descriptors have shape (K_i, 128), where K_i is the keypoint count; None if unavailable.
    descriptors: Optional[np.ndarray]


@dataclass(frozen=True)
class EdgeSpec:
    i: int
    j: int
    edge_type: str
    edge_name: str


@dataclass
class PairMatch_Edge:
    """Matching, RANSAC, and spatial sampling results for candidate edge (i, j)."""

    i: int  # Source image index.
    j: int  # Target image index.
    edge_type: str = "ordinary"  # ordinary / stereo
    edge_name: str = ""  # Semantic name for logging and diagnostics.
    raw_matches: int = 0  # Initial BFMatcher KNN match count.
    ratio_matches: int = 0  # Matches retained by the Lowe ratio test.
    ransac_inliers: int = 0  # Homography RANSAC inlier count.
    selected_matches: int = 0  # Spatially selected matches used in optimization.
    inlier_ratio: float = 0.0  # ransac_inliers / ratio_matches.
    mean_reproj_error: float = math.inf  # Mean RANSAC inlier reprojection error in pixels.
    H_i_to_j: Optional[np.ndarray] = None  # Local homography satisfying p_j ~ H_i_to_j p_i.
    inlier_matches: List[cv2.DMatch] = field(default_factory=list)  # RANSAC inlier matches.
    selected_dmatches: List[cv2.DMatch] = field(
        default_factory=list
    )  # Spatially selected matches for debug drawing.

    # Selected correspondence coordinates in image i.
    selected_pts_i: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float64))

    # Corresponding selected coordinates in image j.
    selected_pts_j: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float64))


@dataclass
class PersistentBlockCache:
    """Original typed observations from the previous successful block for the next two-block

    window.
    """

    keys: List[Tuple[str, int]]
    images: List[ImageRecord]
    valid_edges: List[PairMatch_Edge]


def count_matches_by_type(edges: Sequence[PairMatch_Edge]) -> Tuple[int, int]:
    """Count selected matches on ordinary and stereo edges."""
    return (
        sum(edge.selected_matches for edge in edges if edge.edge_type == "ordinary"),
        sum(edge.selected_matches for edge in edges if edge.edge_type == "stereo"),
    )


def split_edges_by_type(
    edges: Sequence[PairMatch_Edge],
) -> Tuple[List[PairMatch_Edge], List[PairMatch_Edge]]:
    """Partition valid edges explicitly into ordinary and stereo groups."""

    ordinary_edges = [edge for edge in edges if edge.edge_type == "ordinary"]
    stereo_edges = [edge for edge in edges if edge.edge_type == "stereo"]
    return ordinary_edges, stereo_edges
