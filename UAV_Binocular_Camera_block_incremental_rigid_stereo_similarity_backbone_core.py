"""Compatibility entry point for stereo_uav.backbone."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from stereo_uav.backbone import *  # noqa: E402,F403
from stereo_uav.backbone.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
