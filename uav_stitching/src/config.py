"""YAML configuration loading with project-root-relative path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Resolved paths and raw parameter sections for the UAV pipeline."""

    config_path: Path
    project_root: Path
    images_dir: Path
    mrk_file: Path | None
    rtk_file: Path | None
    nav_file: Path | None
    obs_file: Path | None
    cache_dir: Path
    output_dir: Path
    debug_dir: Path
    gps: dict[str, Any]
    raw: dict[str, Any]


def _resolve_optional(project_root: Path, value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    return (project_root / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(config_path: str | Path) -> ProjectConfig:
    """Load a YAML file and resolve all configured paths against the project root.

    The project root is the parent of the ``configs`` directory containing the
    YAML file. This makes CLI behavior independent of the caller's working
    directory and avoids hard-coded machine paths.
    """

    resolved_config = Path(config_path).expanduser().resolve()
    if not resolved_config.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {resolved_config}")
    with resolved_config.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a mapping: {resolved_config}")

    project_root = resolved_config.parent.parent.resolve()
    images_dir = _resolve_optional(project_root, payload.get("images_dir"))
    if images_dir is None:
        raise ValueError("Configuration requires 'images_dir'")

    return ProjectConfig(
        config_path=resolved_config,
        project_root=project_root,
        images_dir=images_dir,
        mrk_file=_resolve_optional(project_root, payload.get("mrk_file")),
        rtk_file=_resolve_optional(project_root, payload.get("rtk_file")),
        nav_file=_resolve_optional(project_root, payload.get("nav_file")),
        obs_file=_resolve_optional(project_root, payload.get("obs_file")),
        cache_dir=_resolve_optional(project_root, payload.get("cache_dir", "cache")) or project_root / "cache",
        output_dir=_resolve_optional(project_root, payload.get("output_dir", "outputs")) or project_root / "outputs",
        debug_dir=_resolve_optional(project_root, payload.get("debug_dir", "debug")) or project_root / "debug",
        gps=dict(payload.get("gps") or {}),
        raw=payload,
    )
