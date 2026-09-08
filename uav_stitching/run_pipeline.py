#!/usr/bin/env python
"""Run one or all stages of the UAV stitching reproduction."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ProjectConfig, load_config  # noqa: E402


STAGES = ("metadata", "pairs", "matches", "affine", "projective", "warp", "blend", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=STAGES, default="all")
    return parser


def _command_for(stage: str, config: ProjectConfig) -> list[str]:
    scripts = PROJECT_ROOT / "scripts"
    if stage == "metadata":
        if config.mrk_file is None:
            raise ValueError("metadata stage requires mrk_file in the configuration")
        command = [
            sys.executable, str(scripts / "inspect_dataset.py"),
            "--images", str(config.images_dir), "--mrk", str(config.mrk_file),
            "--output", str(config.cache_dir / "image_metadata.csv"),
        ]
        for name, value in (("--rtk", config.rtk_file), ("--nav", config.nav_file), ("--obs", config.obs_file)):
            if value is not None:
                command.extend((name, str(value)))
        return command
    if stage == "pairs":
        return [sys.executable, str(scripts / "build_pairs.py"), "--config", str(config.config_path)]
    if stage == "matches":
        return [sys.executable, str(scripts / "match_features.py"), "--config", str(config.config_path)]
    if stage == "affine":
        return [sys.executable, str(scripts / "optimize_affine.py"), "--config", str(config.config_path)]
    if stage == "projective":
        return [sys.executable, str(scripts / "optimize_projective.py"), "--config", str(config.config_path)]
    if stage == "warp":
        return [sys.executable, str(scripts / "evaluate.py"), "--config", str(config.config_path), "--render"]
    if stage == "blend":
        return [sys.executable, str(scripts / "stitch.py"), "--config", str(config.config_path)]
    raise ValueError(f"Unsupported stage: {stage}")


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    requested = ["metadata", "pairs", "matches", "affine", "projective", "warp", "blend"] \
        if args.stage == "all" else [args.stage]
    for index, stage in enumerate(requested, start=1):
        print(f"\n=== Stage {index}/{len(requested)}: {stage} ===", flush=True)
        subprocess.run(_command_for(stage, config), cwd=PROJECT_ROOT, check=True)
    if args.stage in {"blend", "all"}:
        print("\n=== Final evaluation ===", flush=True)
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate.py"), "--config", str(config.config_path)],
            cwd=PROJECT_ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
