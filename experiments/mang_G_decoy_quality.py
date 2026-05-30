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

    # Memory-efficient version: dedup to UNIQUE (lig, decoy_target) pairs
    # before joining with actives. Then map back to decoy_id at the end.
    #
    # If 5.47M decoy rows reduce to ~1.4M unique (lig, target) pairs and
    # active_lig_target is ~820K, the join output is bounded by the number
    # of distinct (lig, decoy_target, active_target) triples — typically a
    # few million, not the cross-product.
    dl_t = decoy_pairs.select(["lig", "decoy_target"]).unique()
    print(f"unique (lig, decoy_target) pairs: {dl_t.height:,}", flush=True)

    # Aggregate active_lig_target per ligand so the join fans out by
    # (number of OTHER targets per ligand) rather than per (decoy_id, lig).
    active_per_lig = (
        active_lig_target.group_by("lig").agg([
            pl.col("active_target").alias("active_targets"),
            pl.col("active_source").first().alias("sample_active_source"),
        ])
    )

    dirty_pairs = (
        dl_t.join(active_per_lig, on="lig", how="inner")
        .with_columns(
            # remove the decoy's own target from the list of "other" targets
            pl.col("active_targets").list.set_difference(
                pl.col("decoy_target").reshape((-1, 1)).cast(pl.List(pl.Utf8))
            ).alias("other_targets")
        )
        .with_columns(pl.col("other_targets").list.len().alias("n_other_targets"))
        .filter(pl.col("n_other_targets") > 0)
        .select([
            "lig", "decoy_target", "n_other_targets",
            pl.col("other_targets").list.first().alias("sample_other_target"),
            "sample_active_source",
        ])
    )
    print(f"dirty (lig, decoy_target) pairs: {dirty_pairs.height:,}", flush=True)

    per_decoy = (
        decoy_pairs.join(dirty_pairs, on=["lig", "decoy_target"], how="inner")
        .select([
            "decoy_id", "decoy_source", "decoy_target",
            "n_other_targets", "sample_other_target",
            pl.col("sample_active_source").alias("sample_other_source"),
        ])
    )
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
