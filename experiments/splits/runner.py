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


# Param sweeps per protocol. 14-config benchmark:
#   8 baselines (Nhóm 1 — off-the-shelf, no paper-specific port required)
#   6 KG (Nhóm 3 — ours, 3 algorithms × 2 axis modes)
#
# DataSAIL / PLINDER / AVE (Nhóm 2 — paper baselines) are deliberately
# excluded: the current ports are simplified fallbacks (datasail is a
# pure random fallback without mmseqs binaries, plinder_style is a BFS
# port without the paper's Louvain communities, ave_wallach is a
# single-pass not the iterative bias optimisation). Including them as
# headline comparisons would misrepresent prior work.
PARAM_SWEEPS: dict[str, list[dict[str, Any]]] = {
    # ---- Nhóm 1 baselines (8 configs) ----
    "random": [{}],
    "random_per_target": [{}],
    "scaffold": [{}],
    "scaffold_generic": [{}],
    "tanimoto_maxmin": [{"T": 0.4}],
    "protein_cluster": [
        {"identity": 30},
        {"identity": 50},
        {"identity": 90},
    ],
    # ---- Nhóm 3 ours: 3 algorithms × 2 axis modes (6 configs) ----
    # `structural` axes = ligand + scaffold (chemistry only, what models
    # actually consume). `strict` axes = + publication + assay (catches
    # SAR-series leak that scaffold doesn't reach). Both intentionally
    # exclude direct example_has_protein, which saturates K=2 within
    # any single-corpus split.
    "kg_kdisjoint": [
        {"K": 2, "axes": "ligand,scaffold"},
        {"K": 2, "axes": "ligand,scaffold,publication,assay"},
    ],
    "kg_maxmin": [
        {"T": 2, "axes": "ligand,scaffold"},
        {"T": 2, "axes": "ligand,scaffold,publication,assay"},
    ],
    "kg_axis_budget": [
        # structural-only: relax pub/assay to 100% so they don't constrain
        {"K": 2, "budget_publication": 1.0, "budget_assay": 1.0},
        # strict: chemistry tight AND pub/assay tight
        {"K": 2},
    ],
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
        rows = [r for r in audit_rows if not r.get("skipped")]
        if rows:
            # Per-corpus audit file to avoid races when multiple sweep
            # processes run in parallel (one per corpus).
            if args.audit_suffix:
                audit_path = output_root / f"audit_summary__{args.audit_suffix}.csv"
            elif len(corpora) == 1:
                audit_path = output_root / f"audit_summary__{corpora[0]}.csv"
            else:
                audit_path = output_root / "audit_summary.csv"
            audit_df = pl.from_dicts(rows)
            if audit_path.exists():
                prev = pl.read_csv(audit_path)
                audit_df = pl.concat([prev, audit_df], how="diagonal_relaxed")
            audit_df = audit_df.unique(
                subset=["corpus", "protocol", "params", "seed"],
                keep="last",
            )
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
    p.add_argument("--audit-suffix", default="",
                   help="override audit filename suffix (else: derived from corpus)")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
