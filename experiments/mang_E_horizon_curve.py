"""Mảng E — Generalization horizon curve.

Question: How does AUROC change as a function of the KG distance from each
test example to its nearest train example?

Plot AUROC(k) where k = "nearest train is within k hops". As k increases,
AUROC should drop toward the random baseline. The point at which AUROC
saturates near 0.5 is the model's *true generalization horizon* — beyond
that, predictions are no better than random.

Outputs
-------
outputs/experiments/mang_E/
  per_test_distance.parquet    (example_id, fold, label, score, hop)
  auroc_by_hop.csv             (corpus, hop, n_test, auroc, ci_lo, ci_hi)
  report.md

CLI
---
python -m experiments.mang_E_horizon_curve \
    --predictions predictions/morgan_rf__litpcba.parquet \
    --output-dir outputs/experiments/mang_E \
    --max-hop 5
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
    load_canonical_kg,
    load_examples,
    load_predictions,
    split_train_test,
)


# Per-axis leak edge sets. Within a single corpus, example_has_protein
# trivially connects every example to every other in 2 hops via shared
# target — that saturates the horizon curve to a single bin. To get a
# meaningful curve we measure distance through the *ligand* axis only
# (every example reaches its train neighbour via shared ligand → scaffold
# → similar ligand → ...) and report it alongside the all-axis distance
# for completeness.
_AXIS_EDGE_SETS: dict[str, tuple[str, ...]] = {
    "ligand_axis": (
        "example_has_ligand",
        "ligand_similar", "ligand_exact", "ligand_parent_exact",
        "ligand_fingerprint_exact", "ligand_scaffold",
    ),
    "all": (
        "example_has_ligand", "example_has_protein",
        "example_from_publication", "example_from_assay",
        "ligand_similar", "ligand_exact", "ligand_parent_exact",
        "ligand_fingerprint_exact", "ligand_scaffold", "protein_in_cluster",
    ),
}
# Back-compat default.
_LEAK_EDGE_TYPES: tuple[str, ...] = _AXIS_EDGE_SETS["all"]


def annotate_distances(
    preds: pl.DataFrame, kg_dir: Path, max_hop: int,
    schema: PredictionSchema | None = None,
    axis: str = "ligand_axis",
) -> pl.DataFrame:
    """Annotate each test prediction with its BFS distance to the nearest
    train item over the given `axis` edge set.

    Defaults to `ligand_axis` to avoid the protein-axis saturation that
    pegs every test to hop=2 in any single-corpus split.
    """
    s = schema or PredictionSchema()
    train_df, test_df = split_train_test(preds, s)
    _, edges = load_canonical_kg(kg_dir)
    etypes = list(_AXIS_EDGE_SETS.get(axis, _LEAK_EDGE_TYPES))
    leak = edges.filter(pl.col("edge_type").is_in(etypes)).select(["src", "dst"])
    train_ids = train_df.select(pl.col(s.example_id).alias("node_id")).unique()
    reached = bfs_distance(train_ids, leak, max_hop=max_hop)
    examples = load_examples(kg_dir).select(["node_id", "source"])
    return (
        test_df.rename({s.example_id: "node_id"})
        .join(reached, on="node_id", how="left")
        .join(examples, on="node_id", how="left")
        .with_columns(pl.lit(axis).alias("axis"))
    )


def horizon_curve(
    annotated: pl.DataFrame, schema: PredictionSchema | None = None
) -> pl.DataFrame:
    """AUROC + 95% CI per (corpus, hop_bucket)."""
    s = schema or PredictionSchema()
    rows = []
    hop_buckets = annotated.with_columns(
        pl.col("hop").fill_null(99).alias("hop_bucket")
    )
    for (corpus, hop), g in hop_buckets.group_by(["source", "hop_bucket"]):
        if isinstance(corpus, tuple):
            corpus = corpus[0]; hop = hop[0] if isinstance(hop, tuple) else hop
        if g.height < 5:
            continue
        scores = g[s.score].to_numpy()
        labels = g[s.label].to_numpy().astype(np.int8)
        ci = bootstrap_auroc_ci(scores, labels, n_boot=500)
        rows.append({
            "source": corpus, "hop": int(hop),
            "n_test": g.height, "n_pos": int((labels == 1).sum()),
            "n_neg": int((labels == 0).sum()),
            "auroc": ci.point, "ci_lo": ci.lower, "ci_hi": ci.upper,
        })
    return pl.DataFrame(rows).sort(["source", "hop"])


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema = PredictionSchema()
    preds = load_predictions(args.predictions, schema)
    annotated_parts: list[pl.DataFrame] = []
    curve_parts: list[pl.DataFrame] = []
    for axis in _AXIS_EDGE_SETS:
        a = annotate_distances(preds, Path(args.kg_dir), args.max_hop, schema, axis=axis)
        c = horizon_curve(a, schema).with_columns(pl.lit(axis).alias("axis"))
        annotated_parts.append(a)
        curve_parts.append(c)
    annotated = pl.concat(annotated_parts, how="vertical_relaxed")
    curve = pl.concat(curve_parts, how="vertical_relaxed")
    annotated.write_parquet(out / "per_test_distance.parquet")
    curve.write_csv(out / "auroc_by_hop.csv")

    lines = ["# Mảng E — Generalization horizon curve", ""]
    n_test = preds.filter(pl.col(schema.fold) == "test").height
    lines.append(f"- test items: {n_test:,}")
    for axis in _AXIS_EDGE_SETS:
        a_sub = annotated.filter(pl.col("axis") == axis)
        c_sub = curve.filter(pl.col("axis") == axis)
        n_reached = int(a_sub.filter(pl.col("hop").is_not_null()).height)
        lines.append("")
        lines.append(f"## axis = {axis}")
        lines.append(f"- reachable within ≤{args.max_hop} hops: "
                     f"{n_reached:,} ({100*n_reached/max(a_sub.height,1):.1f}%)")
        lines.append(c_sub.to_pandas().round(3).to_markdown(index=False))
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}/per_test_distance.parquet")
    print(f"wrote {out}/auroc_by_hop.csv ({curve.height} rows)")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--kg-dir", default=DEFAULT_KG_DIR, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-hop", type=int, default=5)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
