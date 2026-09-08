"""Paper-oriented large-scale UAV image registration and stitching package."""

from __future__ import annotations

import os
import sys
from pathlib import Path


_DLL_DIRECTORY_HANDLES: list[object] = []


def _bootstrap_windows_conda_dll_path() -> None:
    """Make direct ``python.exe`` invocation behave like Conda activation.

    On Windows, NumPy/SciPy BLAS libraries in ``<env>/Library/bin`` are not
    discoverable when the environment's interpreter is called by absolute path
    without ``conda activate``. Missing this path causes an uncatchable native
    crash on matrix operations. Keep the add_dll_directory handle alive for the
    process and also update PATH for libraries that use LoadLibrary directly.
    """

    if os.name != "nt":
        return
    dll_dir = Path(sys.prefix) / "Library" / "bin"
    if not dll_dir.is_dir():
        return
    dll_text = str(dll_dir)
    existing = os.environ.get("PATH", "").split(os.pathsep)
    if dll_text.casefold() not in {item.casefold() for item in existing if item}:
        os.environ["PATH"] = dll_text + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(dll_text))


_bootstrap_windows_conda_dll_path()

__version__ = "0.1.0"
