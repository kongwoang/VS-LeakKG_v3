"""Mảng A — Publication / Assay relational leak audit.

Question: For each test example, does it share a Publication or Assay node
(via `example_from_publication` / `example_from_assay` edges) with any
train example? Split the test set on that flag and compare AUROCs.

Why KG: this is purely relational leakage — two examples linked by the
same ChEMBL paper or BindingDB assay can be far apart in Tanimoto / scaffold
/ sequence space yet still constitute a leak (same SAR series, same author,
same assay condition). No feature-distance method can detect this.

Inputs
------
predictions parquet (PredictionSchema) with example_ids that match KG node_ids.

Outputs
-------
outputs/experiments/mang_A/
  per_example_tags.parquet     (example_id, fold, label, score,
                                share_pub, share_asy, n_pub_train, n_asy_train)
  delta_auroc_pub.csv          (per-corpus Δ-AUROC table for publication)
  delta_auroc_asy.csv          (per-corpus Δ-AUROC table for assay)
  report.md                    (short autotext summary)

CLI
---
python -m experiments.mang_A_publication_audit \
    --predictions predictions/morgan_rf__litpcba.parquet \
    --output-dir outputs/experiments/mang_A
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from .common import (
    DEFAULT_KG_DIR,
    PredictionSchema,
    bootstrap_auroc_ci,
    delta_auroc_test,
    edges_of_types,
    load_examples,
    load_predictions,
    split_train_test,
)


def _examples_sharing_neighbor(
    train_ids: pl.DataFrame, test_ids: pl.DataFrame, ref_edges: pl.DataFrame
) -> pl.DataFrame:
    """Tag each test_id with (share, n_train) — number of train examples
    sharing at least one ref node (Publication/Assay) with this test example.

    `ref_edges` columns: src=example_id, dst=ref_id (Publication or Assay node).
    """
    # Train side: ref node -> list of train examples.
    train_ref = ref_edges.join(
        train_ids.rename({"node_id": "src"}), on="src", how="semi"
    ).select(["src", "dst"]).unique()
    # Count train examples per ref node.
    ref_n_train = train_ref.group_by("dst").len().rename({"len": "n_train"})

    # Test side: each test_id × ref node.
    test_ref = ref_edges.join(
        test_ids.rename({"node_id": "src"}), on="src", how="semi"
    ).select(["src", "dst"]).unique()
    # Sum train counts over all ref nodes a test_id sees.
    test_tagged = (
        test_ref.join(ref_n_train, on="dst", how="left")
        .with_columns(pl.col("n_train").fill_null(0))
        .group_by("src")
        .agg(pl.col("n_train").sum().alias("n_train_via_ref"))
        .rename({"src": "node_id"})
    )
    # Examples with NO ref edge at all are missing here — back-fill with 0.
    out = (
        test_ids.select("node_id")
        .join(test_tagged, on="node_id", how="left")
        .with_columns(pl.col("n_train_via_ref").fill_null(0))
    )
    return out


def tag_predictions(
    preds: pl.DataFrame,
    kg_dir: Path,
    schema: PredictionSchema | None = None,
) -> pl.DataFrame:
    """Annotate predictions with paper/assay share flags. Returns a frame with
    columns (example_id, fold, label, score, source, share_pub, share_asy,
    n_pub_train, n_asy_train)."""
    s = schema or PredictionSchema()
    train_df, test_df = split_train_test(preds, s)
    train_ids = train_df.select(pl.col(s.example_id).alias("node_id")).unique()
    test_ids = test_df.select(pl.col(s.example_id).alias("node_id")).unique()

    pub = edges_of_types(["example_from_publication"], kg_dir).select(["src", "dst"])
    asy = edges_of_types(["example_from_assay"], kg_dir).select(["src", "dst"])

    pub_tag = _examples_sharing_neighbor(train_ids, test_ids, pub).rename(
        {"n_train_via_ref": "n_pub_train"}
    )
    asy_tag = _examples_sharing_neighbor(train_ids, test_ids, asy).rename(
        {"n_train_via_ref": "n_asy_train"}
    )

    examples = load_examples(kg_dir).select(["node_id", "source"])

    tagged_test = (
        test_df.rename({s.example_id: "node_id"})
        .join(pub_tag, on="node_id", how="left")
        .join(asy_tag, on="node_id", how="left")
        .join(examples, on="node_id", how="left")
        .with_columns([
            pl.col("n_pub_train").fill_null(0),
            pl.col("n_asy_train").fill_null(0),
        ])
        .with_columns([
            (pl.col("n_pub_train") > 0).alias("share_pub"),
            (pl.col("n_asy_train") > 0).alias("share_asy"),
        ])
    )
    return tagged_test


def _delta_table(tagged: pl.DataFrame, flag: str, schema: PredictionSchema) -> pl.DataFrame:
    """Per-corpus Δ-AUROC table between `flag=True` and `flag=False` test partitions."""
    s = schema
    rows = []
    for corpus, g in tagged.group_by("source"):
        if isinstance(corpus, tuple):
            corpus = corpus[0]
        leak = g.filter(pl.col(flag))
        clean = g.filter(~pl.col(flag))
        if not leak.height or not clean.height:
            continue
        sa = leak[s.score].to_numpy(); ya = leak[s.label].to_numpy().astype(np.int8)
        sb = clean[s.score].to_numpy(); yb = clean[s.label].to_numpy().astype(np.int8)
        res = delta_auroc_test(sa, ya, sb, yb, n_boot=1000)
        auc_l = bootstrap_auroc_ci(sa, ya, n_boot=500)
        auc_c = bootstrap_auroc_ci(sb, yb, n_boot=500)
        rows.append({
            "source": corpus,
            "n_leak": leak.height, "n_clean": clean.height,
            "auroc_leak": auc_l.point, "auroc_leak_ci_lo": auc_l.lower, "auroc_leak_ci_hi": auc_l.upper,
            "auroc_clean": auc_c.point, "auroc_clean_ci_lo": auc_c.lower, "auroc_clean_ci_hi": auc_c.upper,
            "delta": res["delta"], "delta_ci_lo": res["ci_low"], "delta_ci_hi": res["ci_high"],
            "pvalue": res["pvalue_two_sided"],
        })
    return pl.DataFrame(rows)


def _render_report(tagged: pl.DataFrame, pub_tab: pl.DataFrame, asy_tab: pl.DataFrame) -> str:
    """Plain-text report summarising the audit."""
    lines = ["# Mảng A — Publication / Assay relational leak audit", ""]
    n_total = tagged.height
    n_pub = int(tagged.filter(pl.col("share_pub")).height)
    n_asy = int(tagged.filter(pl.col("share_asy")).height)
    lines.append(f"- test examples scored: {n_total:,}")
    lines.append(f"- share ≥1 ChEMBL/BindingDB publication with train: {n_pub:,} ({100*n_pub/max(n_total,1):.1f}%)")
    lines.append(f"- share ≥1 ChEMBL assay with train: {n_asy:,} ({100*n_asy/max(n_total,1):.1f}%)")
    lines.append("")
    lines.append("## Δ-AUROC per corpus — publication leak")
    lines.append(pub_tab.to_pandas().to_markdown(index=False) if pub_tab.height else "_no corpus had both partitions populated_")
    lines.append("")
    lines.append("## Δ-AUROC per corpus — assay leak")
    lines.append(asy_tab.to_pandas().to_markdown(index=False) if asy_tab.height else "_no corpus had both partitions populated_")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema = PredictionSchema()
    preds = load_predictions(args.predictions, schema)
    tagged = tag_predictions(preds, Path(args.kg_dir), schema)
    tagged.write_parquet(out / "per_example_tags.parquet")

    pub_tab = _delta_table(tagged, "share_pub", schema)
    asy_tab = _delta_table(tagged, "share_asy", schema)
    pub_tab.write_csv(out / "delta_auroc_pub.csv")
    asy_tab.write_csv(out / "delta_auroc_asy.csv")

    (out / "report.md").write_text(_render_report(tagged, pub_tab, asy_tab), encoding="utf-8")
    print(f"wrote {out}/per_example_tags.parquet ({tagged.height} rows)")
    print(f"wrote {out}/delta_auroc_pub.csv ({pub_tab.height} corpora)")
    print(f"wrote {out}/delta_auroc_asy.csv ({asy_tab.height} corpora)")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--kg-dir", default=DEFAULT_KG_DIR, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
