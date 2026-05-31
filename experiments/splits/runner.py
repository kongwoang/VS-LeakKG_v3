"""Full sweep runner — builds + audits every (corpus, protocol, params, seed)
combination and saves artefacts under data/splits/.

Layout:
    data/splits/<corpus>/<protocol>__<params>__seed<n>.parquet
    data/splits/audit_summary.csv

CLI:
    python -m experiments.splits.runner \
        --corpora DEKOIS,BayesBind \
        --seeds 42,43,44 \
        --data-dir data/splits
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from ..common import DEFAULT_KG_DIR, load_examples
from .audit import audit_split
from .base import build_param_str
from .factory import build_split, list_protocols


# Param sweeps per protocol. Each tuple = a separate run.
PARAM_SWEEPS: dict[str, list[dict[str, Any]]] = {
    "random": [{}],
    "scaffold": [{}],
    "tanimoto_maxmin": [{"T": 0.4}],  # single T; canonical sim edges are T≥0.85
    "protein_cluster": [
        {"identity": 30},
        {"identity": 50},
        {"identity": 90},
    ],
    "kg_kdisjoint": [
        {"K": 1},
        {"K": 2},
        {"K": 3},
    ],
    "kg_maxmin": [
        {"T": 2},
        {"T": 3},
    ],
    "kg_axis_budget": [
        {"K": 2},
        {"K": 2, "budget_ligand": 0.01, "budget_publication": 0.01,
         "budget_assay": 0.01, "budget_scaffold": 0.01,
         "budget_protein": 0.50},  # strict
    ],
    "datasail": [{}],
    "plinder_style": [
        {"depth": 1},
        {"depth": 2},
    ],
    "ave_wallach": [{}],
}


_CORPORA = ("DEKOIS", "BayesBind", "BigBind", "DUD-E", "LIT-PCBA")


def _examples_for_corpus(corpus: str, kg_dir: Path) -> pl.DataFrame:
    ex = load_examples(kg_dir)
    return ex.filter(pl.col("source") == corpus).select(["node_id", "label", "source"])


def run_one(
    corpus: str,
    protocol: str,
    params: dict[str, Any],
    seed: int,
    examples: pl.DataFrame,
    kg_dir: Path,
    output_root: Path,
) -> dict:
    out_dir = output_root / corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    param_str = build_param_str(params)
    out_path = out_dir / f"{protocol}__{param_str}__seed{seed}.parquet"
    if out_path.exists():
        return {"skipped": True, "path": str(out_path)}
    t0 = time.perf_counter()
    try:
        res = build_split(protocol, examples, kg_dir, seed=seed, **params)
    except Exception as ex:
        return {
            "corpus": corpus, "protocol": protocol, "params": param_str,
            "seed": seed, "error": f"{type(ex).__name__}: {ex}",
            "wall_s": time.perf_counter() - t0,
        }
    res.folds.with_columns([
        pl.lit(protocol).alias("protocol"),
        pl.lit(seed).alias("seed"),
        pl.lit(param_str).alias("params"),
    ]).write_parquet(out_path)

    audit = audit_split(res.folds, examples, kg_dir)
    audit.update({
        "corpus": corpus, "protocol": protocol, "params": param_str,
        "seed": seed, "wall_s": round(time.perf_counter() - t0, 2),
        "path": str(out_path.relative_to(output_root)),
        "meta": json.dumps(res.meta),
    })
    return audit


def run(args: argparse.Namespace) -> None:
    kg_dir = Path(args.kg_dir)
    output_root = Path(args.data_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    corpora = args.corpora.split(",") if args.corpora else list(_CORPORA)
    seeds = [int(x) for x in args.seeds.split(",")]
    protos = args.protocols.split(",") if args.protocols else list_protocols()

    audit_rows: list[dict] = []
    for corpus in corpora:
        print(f"\n=== corpus = {corpus} ===", flush=True)
        examples = _examples_for_corpus(corpus, kg_dir)
        if not examples.height:
            print(f"  no examples — skip"); continue
        for proto in protos:
            sweep = PARAM_SWEEPS.get(proto, [{}])
            for params in sweep:
                for seed in seeds:
                    print(f"  [{proto} {build_param_str(params)} seed={seed}] ...",
                          end=" ", flush=True)
                    row = run_one(corpus, proto, params, seed, examples,
                                  kg_dir, output_root)
                    if "error" in row:
                        print(f"ERROR: {row['error']}")
                    elif row.get("skipped"):
                        print("skipped (exists)")
                    else:
                        print(f"n_test={row['n_test']} feasible={row['feasible']} "
                              f"({row['wall_s']}s)")
                    audit_rows.append(row)

    if audit_rows:
        # Filter out skipped rows from the summary.
        rows = [r for r in audit_rows if not r.get("skipped")]
        if rows:
            audit_df = pl.from_dicts(rows)
            audit_path = output_root / "audit_summary.csv"
            # Append if exists; else write fresh.
            if audit_path.exists():
                prev = pl.read_csv(audit_path)
                audit_df = pl.concat([prev, audit_df], how="diagonal_relaxed")
            audit_df.write_csv(audit_path)
            print(f"\nwrote {audit_path} ({audit_df.height} rows)")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kg-dir", default=DEFAULT_KG_DIR, type=Path)
    p.add_argument("--data-dir", default="data/splits", type=Path)
    p.add_argument("--corpora", default="",
                   help="comma-separated; default = all 5")
    p.add_argument("--protocols", default="",
                   help="comma-separated; default = all registered")
    p.add_argument("--seeds", default="42,43,44,45,46")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
