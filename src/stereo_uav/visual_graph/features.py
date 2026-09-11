"""Features for the visual graph algorithm."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .geometry import (
    assert_edge_orientation,
    canonical_image_pair,
    key_text,
    warp_points,
)
from .models import (
    Config,
    FeatureRecord,
    ImageEdge,
    ImageKey,
    ImagePair,
    StereoPair,
)


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    image = cv2.imdecode(
        np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    )
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode: {path}")
    data.tofile(path)


def load_stereo_pairs(
    left: Path, right: Path, cfg: Config
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
                raise ValueError(
                    f"Ambiguous duplicate {camera} frame {frame}: {frames[frame].name}, {path.name}"
                )
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


def extract_features(
    key: ImageKey,
    paths: dict[ImageKey, Path],
    feature_cache: dict[ImageKey, FeatureRecord],
    sift,
    extraction_counts: Counter,
) -> FeatureRecord:
    if key not in feature_cache:
        gray = read_image(paths[key], grayscale=True)
        points, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None:
            descriptors = np.empty((0, 128), dtype=np.float32)
        feature_cache[key] = FeatureRecord(
            np.asarray([p.pt for p in points], dtype=np.float64).reshape(-1, 2),
            np.ascontiguousarray(descriptors, dtype=np.float32),
            np.asarray([p.response for p in points]),
            gray.shape,
        )
        extraction_counts[key] += 1
        assert extraction_counts[key] == 1, "SIFT extracted more than once"
    return feature_cache[key]


def point_policy(total_images: int) -> tuple[int, int, int]:
    return (40, 5, 8) if total_images <= 300 else (20, 5, 4)


def select_evenly_distributed_matches(
    points_i: np.ndarray,
    points_j: np.ndarray,
    P: int,
    rows: int,
    cols: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points_i) < P:
        raise ValueError(f"Need {P} inliers, received {len(points_i)}")
    assert rows * cols == P and points_i.shape == points_j.shape
    low, high = points_i.min(axis=0), points_i.max(axis=0)
    normalized = (points_i - low) / np.maximum(high - low, 1e-12)
    cell_x = np.minimum((normalized[:, 0] * cols).astype(int), cols - 1)
    cell_y = np.minimum((normalized[:, 1] * rows).astype(int), rows - 1)
    cells = cell_y * cols + cell_x
    selected = [int(rng.choice(np.flatnonzero(cells == cell))) for cell in np.unique(cells)]
    remaining = np.setdiff1d(np.arange(len(points_i)), selected)
    if len(selected) < P:
        selected.extend(rng.choice(remaining, P - len(selected), replace=False).tolist())
    assert len(selected) == len(set(selected)) == P
    return points_i[selected].copy(), points_j[selected].copy()


def spatial_coverage(points: np.ndarray, shape: tuple[int, int]) -> float:
    return float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))) / (shape[0] * shape[1]))


def mutual_ratio_matches(forward, backward, ratio: float):
    f = {
        m.queryIdx: m.trainIdx
        for pair in forward
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < ratio * n.distance
    }
    b = {
        m.queryIdx: m.trainIdx
        for pair in backward
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < ratio * n.distance
    }
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
        return extract_features(
            key, self.paths, self.feature_cache, self.sift, self.extraction_counts
        )

    def matcher(self, key: ImageKey):
        if key not in self.matchers:
            matcher = cv2.FlannBasedMatcher(
                dict(algorithm=1, trees=self.cfg.FLANN_TREES), dict(checks=self.cfg.FLANN_CHECKS)
            )
            matcher.add([self.feature(key).descriptors])
            matcher.train()
            self.matchers[key] = matcher
        return self.matchers[key]

    def verify(self, key_i: ImageKey, key_j: ImageKey) -> ImageEdge | None:
        return verify_image_pair(key_i, key_j, self)


def verify_image_pair(
    key_i: ImageKey, key_j: ImageKey, verifier: VisualVerifier
) -> ImageEdge | None:
    identity = canonical_image_pair(key_i, key_j)
    if identity in verifier.observations:
        # Canonical identity never reorients the actual observation.
        return verifier.observations[identity]
    cfg = verifier.cfg
    fi, fj = verifier.feature(key_i), verifier.feature(key_j)
    row = dict(
        key_i=key_text(key_i),
        key_j=key_text(key_j),
        raw_matches=0,
        ratio_matches=0,
        mutual_matches=0,
        ransac_inliers=0,
        inlier_ratio=0.0,
        mean_reproj_error=np.nan,
        coverage_i=0.0,
        coverage_j=0.0,
        selected_matches=0,
        verified=False,
        reason="insufficient_descriptors",
    )
    verifier.observations[identity] = None
    verifier.diagnostics.append(row)
    if min(len(fi.descriptors), len(fj.descriptors)) < 2:
        return None
    forward = verifier.matcher(key_j).knnMatch(fi.descriptors, k=2)
    backward = verifier.matcher(key_i).knnMatch(fj.descriptors, k=2)
    ratio, mutual = mutual_ratio_matches(forward, backward, cfg.LOWE_RATIO)
    row.update(
        raw_matches=len(forward),
        ratio_matches=len(ratio),
        mutual_matches=len(mutual),
        reason="insufficient_mutual_matches",
    )
    if len(mutual) < max(4, verifier.P, cfg.MIN_RANSAC_INLIERS):
        return None
    indices = np.asarray(mutual)
    pi, pj = fi.keypoints[indices[:, 0]], fj.keypoints[indices[:, 1]]
    H, mask = cv2.findHomography(
        pi, pj, cv2.RANSAC, cfg.RANSAC_REPROJ_THRESH, maxIters=4000, confidence=0.995
    )
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
    row.update(
        ransac_inliers=count,
        inlier_ratio=inlier_ratio,
        mean_reproj_error=error,
        coverage_i=ci,
        coverage_j=cj,
        reason="quality_gate_failed",
    )
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
    seed = np.random.SeedSequence(
        [cfg.RANDOM_SEED, key_i[1], key_j[1], int(key_i[0] == "right"), int(key_j[0] == "right")]
    )
    si, sj = select_evenly_distributed_matches(
        pi, pj, verifier.P, verifier.rows, verifier.cols, np.random.default_rng(seed)
    )
    quality = min(ci, cj) * inlier_ratio * math.log1p(count) / (error + 1e-6)
    edge = ImageEdge(
        key_i,
        key_j,
        "",
        len(forward),
        len(ratio),
        count,
        inlier_ratio,
        error,
        H,
        pi,
        pj,
        si,
        sj,
        verifier.P,
        ci,
        cj,
        quality,
    )
    assert_edge_orientation(edge)
    row.update(verified=True, reason="accepted", selected_matches=verifier.P)
    verifier.observations[identity] = edge
    return edge
