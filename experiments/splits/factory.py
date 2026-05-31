"""Dispatch / build entry point."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

# Side-effect imports register protocols in PROTOCOLS dict.
from . import baseline_splits as _baselines  # noqa: F401
from . import kg_splits as _kg              # noqa: F401
from . import paper_splits as _papers       # noqa: F401
from .base import PROTOCOLS, SplitResult


def list_protocols() -> list[str]:
    return sorted(PROTOCOLS.keys())


def build_split(
    protocol: str,
    examples: pl.DataFrame,
    kg_dir: Path,
    *,
    seed: int = 42,
    **params: Any,
) -> SplitResult:
    """Build a split using the named protocol.

    Parameters
    ----------
    protocol : one of the registered names (see `list_protocols()`)
    examples : DataFrame with at least (node_id, label, source) columns
    kg_dir   : path to outputs/kg containing canonical_{nodes,edges}.parquet
    seed     : random seed
    **params : protocol-specific tuning parameters

    Returns
    -------
    SplitResult (folds DataFrame + meta dict)
    """
    if protocol not in PROTOCOLS:
        raise KeyError(
            f"unknown protocol '{protocol}'. Available: {sorted(PROTOCOLS)}"
        )
    fn = PROTOCOLS[protocol]
    return fn(examples, kg_dir, seed=seed, **params)
