"""Resolve paths to the v1 dataset and processed graph.

VS-LeakKG_v2 does not ship its own dataset. It reads the raw inputs and the
v1 / v2 processed parquets from the companion v1 repository on disk. By
default we assume the v1 repo is the sibling directory:

    D:/hoangpc/VS-LeakKG          (Windows default)
    ~/VS-LeakKG                   (Linux default if env var unset)

Override either default by exporting `VSLEAKKG_V1_ROOT` before running.

Example:

    >>> from vsleakkg.v2.datapaths import data_root, processed_dir
    >>> data_root()
    PosixPath('/home/user/VS-LeakKG')
    >>> processed_dir()
    PosixPath('/home/user/VS-LeakKG/data/processed')
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ENV_VAR = "VSLEAKKG_V1_ROOT"


def _default_root() -> Path:
    if sys.platform.startswith("win"):
        return Path("D:/hoangpc/VS-LeakKG")
    return Path.home() / "VS-LeakKG"


def data_root() -> Path:
    """Root of the v1 repo (where raw data + processed parquets live)."""
    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return _default_root()


def processed_dir() -> Path:
    """Directory containing the processed mvp2_* and (later) v2_* parquets."""
    return data_root() / "data" / "processed"


def raw_dir() -> Path:
    """Directory containing the raw dataset archives unpacked by the v1 repo."""
    return data_root() / "data" / "raw"


def require_data_root() -> Path:
    """Like `data_root()` but raises if the directory is missing.

    Use this in entry-point scripts that actually need to read data; library
    modules should defer the check to call sites so importing the package
    never fails on a machine without the v1 data.
    """
    root = data_root()
    if not root.exists():
        raise FileNotFoundError(
            f"VS-LeakKG v1 data root not found at {root}. "
            f"Set the {_ENV_VAR} env var to point at your local checkout of "
            f"https://github.com/kongwoang/VS-LeakKG."
        )
    return root
