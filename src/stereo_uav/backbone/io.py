"""Io for the backbone algorithm."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from . import config
from .models import (
    ImageRecord,
)


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """Read an image with OpenCV, including Unicode paths on Windows."""

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    """Write an image with OpenCV, including Unicode paths on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".jpg"
    ok, data = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    data.tofile(str(path))


def to_jsonable_matrix(matrix: Optional[np.ndarray]) -> Optional[List[List[float]]]:
    """Convert a NumPy matrix to a JSON-serializable nested list."""
    if matrix is None:
        return None
    return [[float(value) for value in row] for row in matrix]


def make_output_dirs(output_dir: Path) -> Dict[str, Path]:
    """Create separate directories for mosaics and intermediate data."""
    dirs = {
        "root": output_dir,
        "mosaics": output_dir / "mosaics",
        "data": output_dir / "data",
        "logs": output_dir / "data" / "logs",
        "blocks": output_dir / "data" / "blocks",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def make_block_output_dirs(output_dirs: Dict[str, Path], block_idx: int) -> Dict[str, Path]:
    """Create a per-block directory containing CSV data only."""

    block_root = output_dirs["blocks"] / f"block_{block_idx:04d}"
    dirs = {
        "rigid_stereo_debug": block_root / "rigid_stereo_debug",
        "transforms": block_root / "transforms",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def natural_sort_key(path: Path) -> Tuple[object, ...]:
    """Sort naturally by numerical components of the filename."""
    return tuple(
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)
    )


def extract_frame_number(path: Path) -> int:
    # The stem excludes the filename extension.
    # Match contiguous sequences of digits.
    numbers = re.findall(r"\d+", path.stem)  # Extract numerical filename components.

    # Find every contiguous sequence of digits in the filename stem.

    if not numbers:
        # A missing frame number prevents frame synchronization.
        # Raise an explicit error instead of silently skipping the image.
        raise ValueError(f"Cannot extract frame number from image name: {path.name}")

    return int(numbers[-1])


def collect_image_paths(image_dir: Path) -> List[Path]:
    """Collect and naturally sort image paths for one camera stream."""

    root = Path(image_dir)  # Resolve the image directory.
    if not root.exists():
        raise FileNotFoundError(f"IMAGE_DIR does not exist: {root}")
    paths = [
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=natural_sort_key)  # Return a list of image paths.


def read_one_image(path: Path, idx: int) -> ImageRecord:
    image = imread_unicode(path)
    if image is None:
        raise RuntimeError(f"Unreadable image: {path}")
    # Convert BGR pixels to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape[:2]
    return ImageRecord(idx, path, path.name, image, gray, width, height)


def image_pixels(image: ImageRecord) -> np.ndarray:
    """Return BGR pixels, reloading released historical frames only when rendering requires them."""

    if image.image is not None:
        return image.image

    pixels = imread_unicode(image.path)
    if pixels is None:
        raise RuntimeError(f"Unreadable image: {image.path}")
    return pixels


def load_image_batches(
    left_in: Path,
    right_in: Path,
    batch_size_per_side: int = 2,
) -> Iterator[Tuple[List[ImageRecord], List[ImageRecord]]]:
    """Legacy non-overlapping two-frame batches; the main pipeline uses sliding windows."""

    left_paths = collect_image_paths(left_in)
    right_paths = collect_image_paths(right_in)
    left_by_frame = {extract_frame_number(path): path for path in left_paths}
    right_by_frame = {extract_frame_number(path): path for path in right_paths}
    if len(left_by_frame) != len(left_paths) or len(right_by_frame) != len(right_paths):
        raise RuntimeError("Duplicate frame numbers found in stereo input directories.")

    common_frames = sorted(set(left_by_frame) & set(right_by_frame))
    usable_count = len(common_frames) // batch_size_per_side * batch_size_per_side
    if usable_count < batch_size_per_side:
        raise RuntimeError("Need at least one frame-aligned stereo batch.")

    for start in range(0, usable_count, batch_size_per_side):
        frame_numbers = common_frames[start : start + batch_size_per_side]
        yield (
            [read_one_image(left_by_frame[number], number) for number in frame_numbers],
            [read_one_image(right_by_frame[number], number) for number in frame_numbers],
        )


def load_incremental_image_windows(
    left_in: Path,
    right_in: Path,
) -> Iterator[Tuple[List[ImageRecord], List[ImageRecord]]]:

    left_paths = collect_image_paths(left_in)
    right_paths = collect_image_paths(right_in)

    print(f"Found {len(left_paths)} left images in {left_in}")
    print(f"Found {len(right_paths)} right images in {right_in}")

    left_by_frame = {extract_frame_number(path): path for path in left_paths}
    right_by_frame = {extract_frame_number(path): path for path in right_paths}

    if len(left_by_frame) != len(left_paths):
        raise RuntimeError("Duplicate frame numbers found in left_in.")

    if len(right_by_frame) != len(right_paths):
        raise RuntimeError("Duplicate frame numbers found in right_in.")

    common_frames = sorted(set(left_by_frame) & set(right_by_frame))
    left_only = sorted(set(left_by_frame) - set(right_by_frame))
    right_only = sorted(set(right_by_frame) - set(left_by_frame))

    if left_only:
        print(f"Warning: skip {len(left_only)} left-only frames: {left_only}")

    if right_only:
        print(f"Warning: skip {len(right_only)} right-only frames: {right_only}")

    if len(common_frames) < 2:
        raise RuntimeError("Need at least 2 frame-aligned stereo pairs.")

    # Advance one time step; consecutive blocks share one stereo frame.
    total_windows = len(common_frames) - 1

    previous_frame: Optional[int] = None
    previous_left: Optional[ImageRecord] = None
    previous_right: Optional[ImageRecord] = None

    for window_idx in tqdm(range(total_windows), desc="Loading incremental image windows"):
        frame_numbers = common_frames[window_idx : window_idx + 2]

        first_frame, second_frame = frame_numbers

        # Reuse the previous window's second frame to avoid decoding it again.
        if (
            previous_frame == first_frame
            and previous_left is not None
            and previous_right is not None
        ):
            left_first = previous_left
            right_first = previous_right
        else:
            left_first = read_one_image(left_by_frame[first_frame], first_frame)
            right_first = read_one_image(right_by_frame[first_frame], first_frame)

        left_second = read_one_image(left_by_frame[second_frame], second_frame)
        right_second = read_one_image(right_by_frame[second_frame], second_frame)

        left_batch = [left_first, left_second]
        right_batch = [right_first, right_second]

        previous_frame = second_frame
        previous_left = left_second
        previous_right = right_second

        yield left_batch, right_batch
