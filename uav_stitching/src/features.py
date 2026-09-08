"""Cached OpenCV SIFT feature extraction."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .utils import ensure_parent


@dataclass(frozen=True)
class FeatureSet:
    """Serializable SIFT keypoint attributes and float32 descriptors."""

    image_id: int
    filename: str
    xy: np.ndarray
    size: np.ndarray
    angle: np.ndarray
    response: np.ndarray
    octave: np.ndarray
    class_id: np.ndarray
    descriptors: np.ndarray
    image_width: int
    image_height: int
    nfeatures: int
    source_size: int
    source_mtime_ns: int

    @property
    def count(self) -> int:
        return int(self.xy.shape[0])


def extract_sift(image_id: int, image_path: str | Path, nfeatures: int = 8000) -> FeatureSet:
    """Extract SIFT once from the full-resolution first image frame."""

    path = Path(image_path).resolve()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"OpenCV could not decode image: {path}")
    detector = cv2.SIFT_create(nfeatures=int(nfeatures))
    keypoints, descriptors = detector.detectAndCompute(image, None)
    if descriptors is None or not keypoints:
        descriptors = np.empty((0, 128), dtype=np.float32)
        xy = np.empty((0, 2), dtype=np.float32)
        size = angle = response = np.empty((0,), dtype=np.float32)
        octave = class_id = np.empty((0,), dtype=np.int32)
    else:
        xy = np.asarray([point.pt for point in keypoints], dtype=np.float32)
        size = np.asarray([point.size for point in keypoints], dtype=np.float32)
        angle = np.asarray([point.angle for point in keypoints], dtype=np.float32)
        response = np.asarray([point.response for point in keypoints], dtype=np.float32)
        octave = np.asarray([point.octave for point in keypoints], dtype=np.int32)
        class_id = np.asarray([point.class_id for point in keypoints], dtype=np.int32)
        descriptors = np.asarray(descriptors, dtype=np.float32)
    stat = path.stat()
    return FeatureSet(
        image_id=image_id,
        filename=path.name,
        xy=xy,
        size=size,
        angle=angle,
        response=response,
        octave=octave,
        class_id=class_id,
        descriptors=descriptors,
        image_width=int(image.shape[1]),
        image_height=int(image.shape[0]),
        nfeatures=int(nfeatures),
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
    )


def save_features(path: str | Path, features: FeatureSet) -> None:
    """Atomically cache features in an uncompressed NPZ for fast reload."""

    destination = Path(path)
    ensure_parent(destination)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                schema_version=np.int32(1),
                image_id=np.int32(features.image_id),
                filename=np.asarray(features.filename),
                xy=features.xy,
                size=features.size,
                angle=features.angle,
                response=features.response,
                octave=features.octave,
                class_id=features.class_id,
                descriptors=features.descriptors,
                image_width=np.int32(features.image_width),
                image_height=np.int32(features.image_height),
                nfeatures=np.int32(features.nfeatures),
                source_size=np.int64(features.source_size),
                source_mtime_ns=np.int64(features.source_mtime_ns),
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_features(path: str | Path) -> FeatureSet:
    """Load a feature cache without pickle."""

    cache_path = Path(path)
    with np.load(cache_path, allow_pickle=False) as payload:
        return FeatureSet(
            image_id=int(payload["image_id"]),
            filename=str(payload["filename"]),
            xy=payload["xy"].astype(np.float32, copy=False),
            size=payload["size"].astype(np.float32, copy=False),
            angle=payload["angle"].astype(np.float32, copy=False),
            response=payload["response"].astype(np.float32, copy=False),
            octave=payload["octave"].astype(np.int32, copy=False),
            class_id=payload["class_id"].astype(np.int32, copy=False),
            descriptors=payload["descriptors"].astype(np.float32, copy=False),
            image_width=int(payload["image_width"]),
            image_height=int(payload["image_height"]),
            nfeatures=int(payload["nfeatures"]),
            source_size=int(payload["source_size"]),
            source_mtime_ns=int(payload["source_mtime_ns"]),
        )


def load_or_extract_features(
    image_id: int,
    image_path: str | Path,
    cache_path: str | Path,
    *,
    nfeatures: int,
    force: bool = False,
) -> tuple[FeatureSet, bool]:
    """Return valid cached SIFT data or extract and atomically replace it.

    The boolean return value is true when extraction was performed.
    """

    source = Path(image_path).resolve()
    cache = Path(cache_path)
    if cache.is_file() and not force:
        try:
            features = load_features(cache)
            stat = source.stat()
            if (
                features.image_id == image_id
                and features.filename == source.name
                and features.nfeatures == int(nfeatures)
                and features.source_size == stat.st_size
                and features.source_mtime_ns == stat.st_mtime_ns
            ):
                return features, False
        except (OSError, ValueError, KeyError):
            pass
    features = extract_sift(image_id, source, nfeatures=nfeatures)
    save_features(cache, features)
    return features, True
