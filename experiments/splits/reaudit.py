"""Re-audit existing split parquets without rebuilding them.

Useful when:
  - The build sweep skipped existing files (DEKOIS scenario where all
    splits already exist from a prior run), so no audit_summary was written.
  - We want to re-compute audit metrics with a tweaked algorithm.

CLI:
    python -m experiments.splits.reaudit --data-dir data/splits \
        [--corpora DEKOIS,BayesBind]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from ..common import DEFAULT_KG_DIR, load_examples
from .audit import audit_split


def _examples_for_corpus(corpus: str, kg_dir: Path) -> pl.DataFrame:
    ex = load_examples(kg_dir)
    return ex.filter(pl.col("source") == corpus).select(["node_id", "label", "source"])


def run(args: argparse.Namespace) -> None:
    root = Path(args.data_dir)
    kg_dir = Path(args.kg_dir)
    corpora = args.corpora.split(",") if args.corpora else [
        d.name for d in root.iterdir() if d.is_dir()
    ]
    for corpus in corpora:
        cdir = root / corpus
        if not cdir.is_dir():
            continue
        files = sorted(cdir.glob("*.parquet"))
        if not files:
            continue
        examples = _examples_for_corpus(corpus, kg_dir)
        print(f"\n=== {corpus}: {len(files)} files ===", flush=True)
        rows: list[dict] = []
        for f in files:
            # Parse <protocol>__<params>__seed<n>.parquet
            stem = f.stem
            parts = stem.split("__")
            if len(parts) != 3:
                print(f"  skip (unparseable name): {f.name}"); continue
            protocol, params, seed_token = parts
            try:
                seed = int(seed_token.replace("seed", ""))
            except ValueError:
                print(f"  skip (bad seed): {f.name}"); continue

            t0 = time.perf_counter()
            split = pl.read_parquet(f).select(["node_id", "fold"])
            audit = audit_split(split, examples, kg_dir)
            audit.update({
                "corpus": corpus, "protocol": protocol, "params": params,
                "seed": seed, "wall_s": round(time.perf_counter() - t0, 2),
                "path": str(f.relative_to(root)),
            })
            rows.append(audit)
            print(f"  [{protocol} {params} seed={seed}] "
                  f"n_test={audit['n_test']} feasible={audit['feasible']} "
                  f"({audit['wall_s']}s)", flush=True)
        if rows:
            df = pl.from_dicts(rows)
            out_path = root / f"audit_summary__{corpus}.csv"
            df.write_csv(out_path)
            print(f"wrote {out_path} ({df.height} rows)")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/splits", type=Path)
    p.add_argument("--kg-dir", default=DEFAULT_KG_DIR, type=Path)
    p.add_argument("--corpora", default="",
                   help="comma-separated subset; default = every subdir")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
