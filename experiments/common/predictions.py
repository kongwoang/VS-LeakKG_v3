"""Predictions input contract.

The experiments package never owns model code. Every experiment that needs
model outputs accepts a single parquet conforming to PredictionSchema. This
lets us plug in Morgan-RF, DeepDTA, KG-GNN, or any published model's
exported predictions identically.

Schema:
    example_id : str   — must match `node_id` in canonical_nodes.parquet
    score      : f64   — raw model score (higher = more "active")
    label      : i32   — 0 or 1
    fold       : str   — "train" | "test" | "val"
    model      : str   — optional column identifying which model produced the row

Use `split_train_test()` to pull the (train_ids, test_ids) seeds for the
KG-side analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class PredictionSchema:
    example_id: str = "example_id"
    score: str = "score"
    label: str = "label"
    fold: str = "fold"
    model: str = "model"


def load_predictions(path: str | Path, schema: PredictionSchema | None = None) -> pl.DataFrame:
    """Read a predictions parquet, validate required columns, drop nulls.

    Raises ValueError if required columns are missing.
    """
    s = schema or PredictionSchema()
    df = pl.read_parquet(path)
    required = {s.example_id, s.score, s.label, s.fold}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prediction parquet {path} missing columns: {sorted(missing)}")
    return df.filter(
        pl.col(s.example_id).is_not_null()
        & pl.col(s.score).is_not_null()
        & pl.col(s.label).is_not_null()
    )


def split_train_test(
    df: pl.DataFrame, schema: PredictionSchema | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (train, test) sub-frames keyed by the `fold` column."""
    s = schema or PredictionSchema()
    train = df.filter(pl.col(s.fold) == "train")
    test = df.filter(pl.col(s.fold) == "test")
    return train, test
