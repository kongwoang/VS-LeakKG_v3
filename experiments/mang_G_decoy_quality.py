"""Mảng G — Decoy quality audit.

Question: For each "decoy" in DUD-E / DEKOIS / BayesBind, does the KG
reveal that this molecule is actually an *active* of some other target
(via `example_from_assay` linking the decoy's Ligand to an Assay where
its label is 1 against a different protein)?

If yes, the decoy isn't a clean negative — it carries real binding
information that the model can learn. This attacks the property-matching
assumption of DUD-E/DEKOIS-style decoy generation.

Outputs
-------
outputs/experiments/mang_G/
  decoy_with_real_activity.parquet
      (decoy_id, decoy_target, n_other_active_assays, sample_assay_id,
       sample_other_target)
  decoy_quality_summary.csv
      per-corpus rate of "decoys with real activity elsewhere"
  report.md

CLI
---
python -m experiments.mang_G_decoy_quality \
    --output-dir outputs/experiments/mang_G
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from .common import DEFAULT_KG_DIR, edges_of_types, load_examples


def _ligand_to_assays(kg_dir: Path) -> pl.DataFrame:
    """Edge map Ligand -> Assay via the canonical example_from_assay edges
    composed with example_has_ligand."""
    eha = edges_of_types(["example_from_assay"], kg_dir).select(["src", "dst"]).rename(
        {"src": "ex", "dst": "assay"}
    )
    ehl = edges_of_types(["example_has_ligand"], kg_dir).select(["src", "dst"]).rename(
        {"src": "ex", "dst": "lig"}
    )
    return ehl.join(eha, on="ex", how="inner").select(["lig", "assay"]).unique()


def _example_target_map(kg_dir: Path) -> pl.DataFrame:
    """example -> protein (target)."""
    return edges_of_types(["example_has_protein"], kg_dir).select(["src", "dst"]).rename(
        {"src": "ex", "dst": "target"}
    )


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    kg_dir = Path(args.kg_dir)

    examples = load_examples(kg_dir).select(["node_id", "source", "label"]).rename(
        {"node_id": "ex"}
    )
    ex_target = _example_target_map(kg_dir)
    ex_lig = edges_of_types(["example_has_ligand"], kg_dir).select(["src", "dst"]).rename(
        {"src": "ex", "dst": "lig"}
    )

    # Build (lig, target) pairs for actives — small (~500K) and pre-deduped.
    active_lig_target = (
        examples.filter(pl.col("label") == 1)
        .join(ex_target, on="ex", how="inner")
        .join(ex_lig, on="ex", how="inner")
        .select(["lig", pl.col("target").alias("active_target"),
                 pl.col("source").alias("active_source")])
        .unique()
    )
    print(f"active (lig,target) pairs: {active_lig_target.height:,}", flush=True)

    # Build (decoy_id, lig, target) for every decoy.
    decoy_pairs = (
        examples.filter(pl.col("label") == 0)
        .join(ex_target, on="ex", how="inner")
        .join(ex_lig, on="ex", how="inner")
        .select([pl.col("ex").alias("decoy_id"),
                 pl.col("source").alias("decoy_source"),
                 pl.col("target").alias("decoy_target"),
                 "lig"])
    )
    print(f"decoy (id,lig,target): {decoy_pairs.height:,}", flush=True)

    # The join: for each decoy, look up actives sharing the same ligand
    # but a DIFFERENT target. Pre-aggregating actives per ligand keeps the
    # join from exploding (a ligand active on N targets joins N times, not
    # N × #decoys-with-same-lig as the naive cross-product would).
    decoy_join = (
        decoy_pairs.join(active_lig_target, on="lig", how="inner")
        .filter(pl.col("active_target") != pl.col("decoy_target"))
    )
    print(f"decoy×active intersect (diff target): {decoy_join.height:,}", flush=True)

    per_decoy = (decoy_join.group_by(["decoy_id", "decoy_source", "decoy_target"])
                 .agg([
                     pl.col("active_target").n_unique().alias("n_other_targets"),
                     pl.col("active_target").first().alias("sample_other_target"),
                     pl.col("active_source").first().alias("sample_other_source"),
                 ]))
    # Rename for output compatibility.
    per_decoy = per_decoy.rename({"decoy_source": "source", "decoy_target": "target"})
    decoys = examples.filter(pl.col("label") == 0)  # for summary count
    print(f"per_decoy unique rows: {per_decoy.height:,}", flush=True)
    per_decoy.write_parquet(out / "decoy_with_real_activity.parquet")

    summary = (decoys.group_by("source").agg([
        pl.len().alias("n_decoys"),
    ]).join(
        per_decoy.group_by("source").len().rename({"len": "n_active_elsewhere"}),
        on="source", how="left",
    ).with_columns(
        pl.col("n_active_elsewhere").fill_null(0).alias("n_active_elsewhere"),
    ).with_columns(
        (100 * pl.col("n_active_elsewhere") / pl.col("n_decoys")).round(2).alias("pct_dirty"),
    ).sort("pct_dirty", descending=True))
    summary.write_csv(out / "decoy_quality_summary.csv")

    lines = ["# Mảng G — Decoy quality audit", ""]
    lines.append(f"- total decoys scanned: {decoys.height:,}")
    lines.append(f"- decoys that have real activity on another target: {per_decoy.height:,}")
    lines.append("")
    lines.append("## Per-corpus decoy quality")
    lines.append(summary.to_pandas().to_markdown(index=False))
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}/decoy_with_real_activity.parquet ({per_decoy.height} rows)")
    print(f"wrote {out}/decoy_quality_summary.csv")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kg-dir", default=DEFAULT_KG_DIR, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
