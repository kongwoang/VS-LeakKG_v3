"""Mảng H — Retrospective AUROC decomposition for published predictions.

Question: For a published VS model's reported AUROC on (say) DUD-E or
DEKOIS, what fraction of the score comes from "clean" test items vs.
"leaky" items (within K hops of the train set on the KG)?

This produces a "deserved AUROC" estimate by restricting evaluation to
KG-clean test items. The gap to the originally claimed AUROC quantifies
the inflation due to leakage.

Inputs
------
predictions parquet (PredictionSchema) — must have train+test rows so we
know which examples were the published model's train set. For papers
that don't release train/test assignments explicitly, pass --train-source
to mark a whole corpus (e.g. ChEMBL) as the implicit training pool.

Outputs
-------
outputs/experiments/mang_H/
  decomposition_<model>.csv     (corpus, partition, n, auroc, ci_lo, ci_hi)
  delta_per_corpus.csv          (corpus, claimed, deserved, delta)
  report.md

CLI
---
python -m experiments.mang_H_retrospective \
    --predictions predictions/deepdta__dude.parquet \
    --output-dir outputs/experiments/mang_H \
    --k-hop 2
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


def _build_train_anchor(
    preds: pl.DataFrame,
    kg_dir: Path,
    train_source: str | None,
    schema: PredictionSchema,
) -> pl.DataFrame:
    """Pick the training anchor set:
      - if predictions has a fold='train' partition, use that;
      - else if --train-source given, use every Example from that corpus;
      - else raise.
    """
    s = schema
    train_df = preds.filter(pl.col(s.fold) == "train")
    if train_df.height:
        return train_df.select(pl.col(s.example_id).alias("node_id")).unique()
    if train_source:
        ex = load_examples(kg_dir).filter(pl.col("source") == train_source)
        if not ex.height:
            raise SystemExit(f"--train-source={train_source} matched 0 examples")
        return ex.select("node_id")
    raise SystemExit(
        "predictions parquet has no fold='train' rows; pass --train-source <CORPUS>"
    )


def decompose(
    preds: pl.DataFrame,
    kg_dir: Path,
    k_hop: int,
    train_source: str | None,
    schema: PredictionSchema | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    s = schema or PredictionSchema()
    _, test_df = split_train_test(preds, s)
    if not test_df.height:
        # All rows are test (no fold split provided) — treat them all as test
        # and rely on --train-source for the anchor.
        test_df = preds

    train_seeds = _build_train_anchor(preds, kg_dir, train_source, s)

    _, edges = load_canonical_kg(kg_dir)
    leak = edges.filter(pl.col("edge_type").is_in(list(_LEAK_EDGE_TYPES))).select(["src", "dst"])
    reached = bfs_distance(train_seeds, leak, max_hop=k_hop)
    examples = load_examples(kg_dir).select(["node_id", "source"])
    annotated = (
        test_df.rename({s.example_id: "node_id"})
        .join(reached, on="node_id", how="left")
        .join(examples, on="node_id", how="left")
        .with_columns(pl.col("hop").is_not_null().alias("is_leak"))
    )

    rows = []
    delta_rows = []
    for corpus, g in annotated.group_by("source"):
        if isinstance(corpus, tuple):
            corpus = corpus[0]
        for partition, mask in [("all", pl.lit(True)),
                                ("leak", pl.col("is_leak")),
                                ("clean", ~pl.col("is_leak"))]:
            sub = g.filter(mask)
            if sub.height < 5:
                continue
            sc = sub[s.score].to_numpy()
            la = sub[s.label].to_numpy().astype(np.int8)
            ci = bootstrap_auroc_ci(sc, la, n_boot=500)
            rows.append({
                "corpus": corpus, "partition": partition, "n": sub.height,
                "auroc": ci.point, "ci_lo": ci.lower, "ci_hi": ci.upper,
            })
        # claimed vs deserved
        claim = next((r for r in rows if r["corpus"] == corpus and r["partition"] == "all"), None)
        deserved = next((r for r in rows if r["corpus"] == corpus and r["partition"] == "clean"), None)
        if claim and deserved:
            delta_rows.append({
                "corpus": corpus,
                "claimed_auroc": claim["auroc"],
                "deserved_auroc": deserved["auroc"],
                "delta_inflation": claim["auroc"] - deserved["auroc"],
                "n_claim": claim["n"], "n_clean": deserved["n"],
            })
    return pl.DataFrame(rows), pl.DataFrame(delta_rows)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema = PredictionSchema()
    preds = load_predictions(args.predictions, schema)
    decomp, delta = decompose(preds, Path(args.kg_dir), args.k_hop, args.train_source, schema)
    decomp.write_csv(out / "decomposition.csv")
    delta.write_csv(out / "delta_per_corpus.csv")

    lines = ["# Mảng H — Retrospective AUROC decomposition", ""]
    lines.append(f"- k-hop leak threshold: {args.k_hop}")
    lines.append(f"- training anchor: "
                 f"{'fold=train rows' if preds.filter(pl.col(schema.fold)=='train').height else args.train_source}")
    lines.append("")
    lines.append("## Decomposition")
    lines.append(decomp.to_pandas().round(3).to_markdown(index=False))
    lines.append("")
    lines.append("## Inflation summary")
    lines.append(delta.to_pandas().round(3).to_markdown(index=False)
                 if delta.height else "_no corpus had all three partitions populated_")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}/decomposition.csv ({decomp.height} rows)")
    print(f"wrote {out}/delta_per_corpus.csv ({delta.height} rows)")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--kg-dir", default=DEFAULT_KG_DIR, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--k-hop", type=int, default=2,
                   help="example is 'leak' if within k hops of any train anchor")
    p.add_argument("--train-source", default=None,
                   help="corpus to treat as the implicit training pool when predictions "
                        "has no fold='train' rows (e.g. 'BigBind')")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
