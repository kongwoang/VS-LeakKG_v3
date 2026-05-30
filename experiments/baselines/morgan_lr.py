"""Morgan-FP Logistic Regression baseline.

Mirrors `morgan_rf.py` but trains an sklearn LogisticRegression with L2
regularisation. Useful as a faster sanity check (a few seconds vs minutes
for RF) and as a linear-vs-nonlinear comparison point.

CLI mirrors morgan_rf.py.
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
    C: float = 1.0,
    train_cap: int = 50000,
    n_bits: int = 2048,
    radius: int = 2,
    seed: int = 42,
    max_iter: int = 1000,
) -> pl.DataFrame:
    from sklearn.linear_model import LogisticRegression

    train = hydrated.filter(pl.col("fold") == "train")
    if not train.height:
        raise SystemExit("hydrated split has 0 train rows")
    if train.height > train_cap:
        train = train.sample(n=train_cap, seed=seed, with_replacement=False)
    print(f"train n={train.height}", flush=True)

    y_train = train["label"].to_numpy().astype(int)
    if len(set(y_train.tolist())) < 2:
        return hydrated.with_columns(pl.lit(0.5, dtype=pl.Float64).alias("score"))

    X_train, used_rdkit = morgan_fingerprints(
        train["smiles"].to_list(), n_bits=n_bits, radius=radius
    )
    print(f"featurised train: shape={X_train.shape}, rdkit={used_rdkit}", flush=True)
    clf = LogisticRegression(
        C=C, max_iter=max_iter, solver="liblinear", random_state=seed
    )
    clf.fit(X_train, y_train)

    # Batched scoring (see morgan_rf.py for rationale on large corpora).
    batch_size = 100_000
    all_smiles = hydrated["smiles"].to_list()
    n = len(all_smiles)
    scores_chunks: list[np.ndarray] = []
    for i in range(0, n, batch_size):
        batch = all_smiles[i:i + batch_size]
        X_batch, _ = morgan_fingerprints(batch, n_bits=n_bits, radius=radius)
        s = clf.predict_proba(X_batch)[:, 1].astype(np.float64)
        scores_chunks.append(s)
    scores = np.concatenate(scores_chunks)
    return hydrated.with_columns(pl.Series("score", scores))


def run(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    split = pl.read_parquet(args.split)
    print(f"loaded split: {split.height} rows", flush=True)
    hydrated = hydrate_split(split, Path(args.kg_dir))
    print(f"hydrated: {hydrated.height} rows", flush=True)
    scored = train_and_score(
        hydrated, C=args.C, train_cap=args.train_cap,
        n_bits=args.n_bits, seed=args.seed,
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
    p.add_argument("-C", type=float, default=1.0, dest="C")
    p.add_argument("--train-cap", type=int, default=50000)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-name", default="morgan_lr")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
