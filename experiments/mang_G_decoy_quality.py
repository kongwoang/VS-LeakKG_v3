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
    # Decoy view: corpus example with label == 0.
    decoys = (examples.filter(pl.col("label") == 0)
              .join(ex_target, on="ex", how="left"))
    actives = (examples.filter(pl.col("label") == 1)
               .join(ex_target, on="ex", how="left"))

    lig_assay = _ligand_to_assays(kg_dir)
    ex_lig = edges_of_types(["example_has_ligand"], kg_dir).select(["src", "dst"]).rename(
        {"src": "ex", "dst": "lig"}
    )

    decoy_with_lig = decoys.join(ex_lig, on="ex", how="left")
    active_with_lig = actives.join(ex_lig, on="ex", how="left")

    # An active is "real activity elsewhere" if its ligand appears in an
    # example whose target ≠ the decoy's target AND the example is labelled
    # active.
    other_active = (active_with_lig.select(
        ["lig", pl.col("target").alias("other_target"),
         pl.col("source").alias("other_source")]
    ).unique())

    decoy_join = (decoy_with_lig.join(other_active, on="lig", how="left")
                  .filter(pl.col("other_target").is_not_null()
                          & (pl.col("other_target") != pl.col("target"))))

    per_decoy = (decoy_join.group_by(["ex", "source", "target"])
                 .agg([
                     pl.col("other_target").n_unique().alias("n_other_targets"),
                     pl.col("other_target").first().alias("sample_other_target"),
                     pl.col("other_source").first().alias("sample_other_source"),
                 ])
                 .rename({"ex": "decoy_id", "target": "decoy_target"}))
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
