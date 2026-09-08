"""Shared filesystem, logging, and ordering utilities."""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
_NUMBER_RE = re.compile(r"(\d+)")


def configure_logging(level: str = "INFO") -> None:
    """Configure concise console logging for CLI entry points."""

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def natural_sort_key(value: str | Path) -> tuple[Any, ...]:
    """Return a case-insensitive key that sorts embedded digit runs numerically."""

    text = str(value)
    return tuple(int(part) if part.isdigit() else part.casefold() for part in _NUMBER_RE.split(text))


def find_jpeg_images(images_dir: Path) -> list[Path]:
    """Find JPG/JPEG files directly below *images_dir* in natural filename order."""

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")
    images = [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg"}
    ]
    return sorted(images, key=lambda path: natural_sort_key(path.name))


def ensure_parent(path: Path) -> None:
    """Create the parent directory of *path* if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)


def _temporary_sibling(path: Path) -> Path:
    ensure_parent(path)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    return Path(name)


def write_csv_atomic(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Write a CSV through a same-directory temporary file, then replace atomically."""

    temporary = _temporary_sibling(path)
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write UTF-8 JSON through a temporary file, then replace atomically."""

    temporary = _temporary_sibling(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
