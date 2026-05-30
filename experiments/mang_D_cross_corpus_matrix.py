"""Mảng D — Cross-corpus contamination matrix.

Question: For every (train_corpus, test_corpus) pair, what fraction of the
test corpus's examples is reachable within K hops of the train corpus's
examples on the KG?

Output is a 5×5 matrix that quantifies how much "leak transfer" can occur
between any two benchmarks. Crucially, the same node_ids (ligands, papers,
proteins) span all five corpora in our canonical KG, so this analysis is
*only* possible because of the unified KG.

No predictions are needed — this is pure structural analysis.

Outputs
-------
outputs/experiments/mang_D/
  contamination_matrix.csv     5×5 matrix of pct_leak (k=1, 2, 3)
  per_pair_examples.parquet    (train_corpus, test_corpus, k_hop, n_test,
                                n_leak, pct_leak, sample_path)
  report.md

CLI
---
python -m experiments.mang_D_cross_corpus_matrix \
    --output-dir outputs/experiments/mang_D \
    --k-hops 1,2,3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from .common import (
    DEFAULT_KG_DIR,
    bfs_distance,
    load_canonical_kg,
    load_examples,
)


_LEAK_EDGE_TYPES: tuple[str, ...] = (
    "example_has_ligand",
    "example_has_protein",
    "example_from_publication",
    "example_from_assay",
    "ligand_similar",
    "ligand_exact",
    "ligand_parent_exact",
    "ligand_fingerprint_exact",
    "ligand_scaffold",
    "protein_in_cluster",
)


def cross_corpus_leak(
    examples: pl.DataFrame, leak_edges: pl.DataFrame, k_hops: tuple[int, ...]
) -> pl.DataFrame:
    """For each (train_corpus, test_corpus) pair compute leak% per k."""
    corpora = sorted(examples["source"].drop_nulls().unique().to_list())
    rows = []
    max_h = max(k_hops)
    for train_c in corpora:
        train_seeds = examples.filter(pl.col("source") == train_c).select("node_id")
        if not train_seeds.height:
            continue
        reached = bfs_distance(train_seeds, leak_edges, max_hop=max_h)
        # Use only test examples that are NOT in the train corpus.
        for test_c in corpora:
            if test_c == train_c:
                continue
            test_set = examples.filter(pl.col("source") == test_c).select("node_id")
            if not test_set.height:
                continue
            hit = test_set.join(reached, on="node_id", how="left")
            for k in k_hops:
                n_leak = int(hit.filter(pl.col("hop").is_not_null() & (pl.col("hop") <= k)).height)
                rows.append({
                    "train_corpus": train_c,
                    "test_corpus": test_c,
                    "k_hop": k,
                    "n_test": test_set.height,
                    "n_leak": n_leak,
                    "pct_leak": 100 * n_leak / test_set.height,
                })
    return pl.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    kg_dir = Path(args.kg_dir)
    examples = load_examples(kg_dir).select(["node_id", "source"])
    _, edges = load_canonical_kg(kg_dir)
    leak_edges = edges.filter(pl.col("edge_type").is_in(list(_LEAK_EDGE_TYPES))).select(
        ["src", "dst"]
    )
    k_hops = tuple(int(x) for x in args.k_hops.split(","))

    matrix = cross_corpus_leak(examples, leak_edges, k_hops)
    matrix.write_parquet(out / "per_pair_examples.parquet")

    for k in k_hops:
        wide = matrix.filter(pl.col("k_hop") == k).pivot(
            index="train_corpus", on="test_corpus", values="pct_leak"
        )
        wide.write_csv(out / f"contamination_matrix_k{k}.csv")

    n_corpora = matrix["train_corpus"].n_unique() if matrix.height else 0
    lines = ["# Mảng D — Cross-corpus contamination matrix", ""]
    lines.append(f"- corpora: {n_corpora}")
    lines.append(f"- k-hops measured: {k_hops}")
    lines.append("")
    for k in k_hops:
        wide = matrix.filter(pl.col("k_hop") == k).pivot(
            index="train_corpus", on="test_corpus", values="pct_leak"
        )
        lines.append(f"## % test examples reachable within k={k} hops")
        lines.append(wide.to_pandas().round(1).to_markdown())
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}/per_pair_examples.parquet ({matrix.height} rows)")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kg-dir", default=DEFAULT_KG_DIR, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--k-hops", default="1,2,3",
                   help="comma-separated hop thresholds, e.g. '1,2,3'")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
