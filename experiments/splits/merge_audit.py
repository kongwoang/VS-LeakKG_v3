"""Merge per-corpus audit_summary__*.csv files into one audit_summary.csv.

CLI:
    python -m experiments.splits.merge_audit --data-dir data/splits
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def run(args: argparse.Namespace) -> None:
    root = Path(args.data_dir)
    parts = sorted(root.glob("audit_summary__*.csv"))
    if not parts:
        print(f"no audit_summary__*.csv files under {root}")
        return
    dfs = []
    for p in parts:
        try:
            dfs.append(pl.read_csv(p))
        except pl.exceptions.NoDataError:
            print(f"  skipping empty: {p.name}")
    if not dfs:
        return
    merged = pl.concat(dfs, how="diagonal_relaxed").unique(
        subset=["corpus", "protocol", "params", "seed"], keep="last"
    )
    out = root / "audit_summary.csv"
    merged.write_csv(out)
    print(f"merged {len(dfs)} files → {out} ({merged.height} rows)")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/splits", type=Path)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
