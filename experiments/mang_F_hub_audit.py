"""Mảng F — Hub-driven leak audit.

Question: Do test examples that are *close to a hub node* (a Protein or
Scaffold with degree > HubMitigationConfig.degree_cap = 1000) exhibit
systematically higher AUROC than test examples that are not?

If yes, benchmarks are biased by "bestseller" proteins (kinases, GPCRs)
and scaffolds (benzofuran, indole) — the model learns the pattern of the
hub and is rewarded on all tests touching it.

Outputs
-------
outputs/experiments/mang_F/
  per_test_hub_distance.parquet   (example_id, label, score, hop_to_hub)
  hub_auroc_split.csv             (corpus, partition, n, auroc, ci)
  report.md

CLI
---
python -m experiments.mang_F_hub_audit \
    --predictions predictions/morgan_rf__bigbind.parquet \
    --output-dir outputs/experiments/mang_F \
    --hub-distance 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from .common import (
    DEFAULT_KG_DIR,
    PredictionSchema,
    bfs_distance,
    bootstrap_auroc_ci,
    delta_auroc_test,
    load_canonical_kg,
    load_examples,
    load_predictions,
    split_train_test,
)


def _hub_seeds(kg_dir: Path) -> pl.DataFrame:
    """Return all node_ids flagged is_hub=True by consolidate."""
    nodes, _ = load_canonical_kg(kg_dir)
    return nodes.filter(pl.col("is_hub") == True).select("node_id")  # noqa: E712


def annotate_distance_to_hub(
    preds: pl.DataFrame, kg_dir: Path, max_hop: int,
    schema: PredictionSchema | None = None,
) -> pl.DataFrame:
    s = schema or PredictionSchema()
    _, test_df = split_train_test(preds, s)
    _, edges = load_canonical_kg(kg_dir)
    leak = edges.filter(pl.col("edge_type").is_in([
        "example_has_ligand", "example_has_protein",
        "ligand_scaffold", "ligand_similar",
        "protein_in_cluster",
    ])).select(["src", "dst"])
    hubs = _hub_seeds(kg_dir)
    if not hubs.height:
        return test_df.with_columns(pl.lit(None, dtype=pl.Int64).alias("hop_to_hub"))
    reached = bfs_distance(hubs, leak, max_hop=max_hop).rename({"hop": "hop_to_hub"})
    examples = load_examples(kg_dir).select(["node_id", "source"])
    return (
        test_df.rename({s.example_id: "node_id"})
        .join(reached, on="node_id", how="left")
        .join(examples, on="node_id", how="left")
    )


def partition_auroc(
    annotated: pl.DataFrame, hub_dist: int,
    schema: PredictionSchema | None = None,
) -> pl.DataFrame:
    """Per-corpus AUROC comparison: within hub_dist of a hub vs. not."""
    s = schema or PredictionSchema()
    rows = []
    for corpus, g in annotated.group_by("source"):
        if isinstance(corpus, tuple):
            corpus = corpus[0]
        near = g.filter(pl.col("hop_to_hub").is_not_null() & (pl.col("hop_to_hub") <= hub_dist))
        far = g.filter(pl.col("hop_to_hub").is_null() | (pl.col("hop_to_hub") > hub_dist))
        if not (near.height and far.height):
            continue
        sn = near[s.score].to_numpy(); yn = near[s.label].to_numpy().astype(np.int8)
        sf = far[s.score].to_numpy(); yf = far[s.label].to_numpy().astype(np.int8)
        cn = bootstrap_auroc_ci(sn, yn, n_boot=500)
        cf = bootstrap_auroc_ci(sf, yf, n_boot=500)
        delta = delta_auroc_test(sn, yn, sf, yf, n_boot=1000)
        rows.append({
            "source": corpus,
            "n_near_hub": near.height, "n_far_hub": far.height,
            "auroc_near": cn.point, "auroc_near_lo": cn.lower, "auroc_near_hi": cn.upper,
            "auroc_far": cf.point, "auroc_far_lo": cf.lower, "auroc_far_hi": cf.upper,
            "delta": delta["delta"],
            "delta_ci_lo": delta["ci_low"], "delta_ci_hi": delta["ci_high"],
            "pvalue": delta["pvalue_two_sided"],
        })
    return pl.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema = PredictionSchema()
    preds = load_predictions(args.predictions, schema)
    annotated = annotate_distance_to_hub(preds, Path(args.kg_dir),
                                          max_hop=args.hub_distance + 1, schema=schema)
    annotated.write_parquet(out / "per_test_hub_distance.parquet")
    tab = partition_auroc(annotated, args.hub_distance, schema)
    tab.write_csv(out / "hub_auroc_split.csv")

    lines = ["# Mảng F — Hub-leak audit", ""]
    lines.append(f"- hub-distance threshold: {args.hub_distance}")
    lines.append(f"- test items: {annotated.height:,}")
    n_near = int(annotated.filter(
        pl.col("hop_to_hub").is_not_null() & (pl.col("hop_to_hub") <= args.hub_distance)
    ).height)
    lines.append(f"- within {args.hub_distance} hops of a hub: {n_near:,} "
                 f"({100*n_near/max(annotated.height,1):.1f}%)")
    lines.append("")
    lines.append("## Per-corpus AUROC partition")
    lines.append(tab.to_pandas().round(3).to_markdown(index=False) if tab.height
                 else "_no corpus had both partitions populated_")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}/per_test_hub_distance.parquet ({annotated.height})")
    print(f"wrote {out}/hub_auroc_split.csv ({tab.height} corpora)")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--kg-dir", default=DEFAULT_KG_DIR, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--hub-distance", type=int, default=2,
                   help="treat a test example as 'near a hub' if hop ≤ this")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
