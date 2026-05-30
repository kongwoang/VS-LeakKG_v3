"""Morgan-FP Random Forest baseline.

Reads a split parquet (node_id, fold), hydrates SMILES from the canonical
KG, featurises with Morgan/ECFP4 (radius 2, 2048 bits), trains a sklearn
RandomForestClassifier on the train fold, then scores every row.

Outputs a PredictionSchema parquet (example_id, score, label, fold, model).

CLI
---
python -m experiments.baselines.morgan_rf \
    --split outputs/experiments/mang_C/split_random__DEKOIS.parquet \
    --output predictions/morgan_rf__random__DEKOIS.parquet \
    --train-cap 15000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from ..common import DEFAULT_KG_DIR
from .featurize import morgan_fingerprints
from .hydrate import hydrate_split


def train_and_score(
    hydrated: pl.DataFrame,
    *,
    n_estimators: int = 100,
    train_cap: int = 15000,
    n_bits: int = 2048,
    radius: int = 2,
    seed: int = 42,
) -> pl.DataFrame:
    """Train Morgan-RF on fold='train', score every row. Returns the input
    frame with an added `score` column."""
    from sklearn.ensemble import RandomForestClassifier

    train = hydrated.filter(pl.col("fold") == "train")
    if not train.height:
        raise SystemExit("hydrated split has 0 train rows")
    if train.height > train_cap:
        train = train.sample(n=train_cap, seed=seed, with_replacement=False)
    print(f"train n={train.height}, pos={int((train['label']==1).sum())}, neg={int((train['label']==0).sum())}", flush=True)

    y_train = train["label"].to_numpy().astype(int)
    if len(set(y_train.tolist())) < 2:
        # Degenerate — fill 0.5 everywhere.
        return hydrated.with_columns(pl.lit(0.5, dtype=pl.Float64).alias("score"))

    X_train, used_rdkit = morgan_fingerprints(
        train["smiles"].to_list(), n_bits=n_bits, radius=radius
    )
    print(f"featurised train: shape={X_train.shape}, rdkit={used_rdkit}", flush=True)

    clf = RandomForestClassifier(
        n_estimators=n_estimators, n_jobs=-1, random_state=seed
    )
    clf.fit(X_train, y_train)

    X_all, _ = morgan_fingerprints(
        hydrated["smiles"].to_list(), n_bits=n_bits, radius=radius
    )
    scores = clf.predict_proba(X_all)[:, 1].astype(np.float64)
    return hydrated.with_columns(pl.Series("score", scores))


def run(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    split = pl.read_parquet(args.split)
    if "node_id" not in split.columns or "fold" not in split.columns:
        raise SystemExit(f"split parquet must have (node_id, fold); got {split.columns}")
    print(f"loaded split: {split.height} rows", flush=True)

    hydrated = hydrate_split(split, Path(args.kg_dir))
    n_train = int(hydrated.filter(pl.col("fold") == "train").height)
    n_test = int(hydrated.filter(pl.col("fold") == "test").height)
    print(f"hydrated: {hydrated.height} rows ({n_train} train / {n_test} test)", flush=True)

    scored = train_and_score(
        hydrated,
        n_estimators=args.n_estimators,
        train_cap=args.train_cap,
        n_bits=args.n_bits,
        seed=args.seed,
    )
    out_df = scored.select(["example_id", "score", "label", "fold"]).with_columns(
        pl.lit(args.model_name).alias("model")
    )
    out_df.write_parquet(out)
    print(f"wrote {out} ({out_df.height} rows)", flush=True)


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kg-dir", type=Path, default=DEFAULT_KG_DIR)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--train-cap", type=int, default=15000)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-name", default="morgan_rf")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
