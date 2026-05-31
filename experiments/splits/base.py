"""Shared types + registry for split protocols."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import polars as pl


@dataclass(frozen=True)
class SplitResult:
    """Output of any protocol's `build_split()`.

    Attributes
    ----------
    folds : DataFrame with columns (node_id, fold) at minimum.
            Optionally (leak_mask, group_id, params_str).
    meta  : protocol-specific notes (rejection reasons, n iterations, etc.).
    """
    folds: pl.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return int(self.folds.filter(pl.col("fold") == "train").height)

    @property
    def n_test(self) -> int:
        return int(self.folds.filter(pl.col("fold") == "test").height)

    @property
    def n_val(self) -> int:
        return int(self.folds.filter(pl.col("fold") == "val").height)


# Protocol registry: name -> callable. Registered via @register_protocol decorator.
PROTOCOLS: dict[str, Callable] = {}


def register_protocol(name: str) -> Callable:
    """Decorator: register a protocol implementation under `name`.

    The decorated function must accept:
        build(examples: pl.DataFrame, kg_dir: Path, *, seed: int, **params) -> SplitResult
    """
    def wrap(fn: Callable) -> Callable:
        if name in PROTOCOLS:
            raise ValueError(f"protocol '{name}' already registered")
        PROTOCOLS[name] = fn
        return fn
    return wrap


# Common helpers --------------------------------------------------------------


def stratified_random_assign(
    examples: pl.DataFrame,
    *,
    test_ratio: float = 0.15,
    val_ratio: float = 0.0,
    seed: int = 42,
    by: str = "label",
) -> pl.DataFrame:
    """Stratified random assignment of train/val/test by `by` column.

    Returns DataFrame (node_id, fold) preserving input row count.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    parts = []
    for grp_key, g in examples.group_by(by):
        n = g.height
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        fold = np.array(["train"] * n, dtype=object)
        fold[idx[:n_test]] = "test"
        fold[idx[n_test:n_test + n_val]] = "val"
        parts.append(g.with_columns(pl.Series("fold", fold)))
    return pl.concat(parts, how="vertical_relaxed").select(["node_id", "fold"])


def build_param_str(params: dict[str, Any]) -> str:
    """Stable string encoding of params for filenames, e.g. 'T0.4_K2'."""
    if not params:
        return "default"
    parts = []
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            parts.append(f"{k}{v:g}")
        else:
            parts.append(f"{k}{v}")
    return "_".join(parts)
