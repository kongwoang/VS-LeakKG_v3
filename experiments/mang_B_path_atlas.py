"""Mảng B — Path-typed mispredict atlas.

Question: When the model gets a test example wrong, what is the *shortest
KG path* from that test example to its nearest train example, and what
edge-type signature does the path have? Categorise these path signatures
(e.g. "[ligand_similar]", "[ligand_scaffold, example_from_publication]")
and report their frequency per corpus.

Why KG: feature-distance only detects 1-hop ligand-ligand neighbours.
The KG exposes multi-hop chains through publication, assay, scaffold,
protein-cluster — and tells you *which* chain caused the leak.

Outputs
-------
outputs/experiments/mang_B/
  per_mispredict_path.parquet   one row per mispredicted test_id with
                                its 2-hop path signature
  path_signature_counts.csv     per-corpus × signature frequency
  report.md                     summary + top signatures

CLI
---
python -m experiments.mang_B_path_atlas \
    --predictions predictions/morgan_rf__litpcba.parquet \
    --output-dir outputs/experiments/mang_B \
    --max-hop 4 \
    --mispredict-threshold 0.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from .common import (
    DEFAULT_KG_DIR,
    PredictionSchema,
    load_canonical_kg,
    load_examples,
    load_predictions,
    split_train_test,
)


# Edge types we walk when looking for leak paths. We deliberately exclude
# `example_from_source` (every example links to its corpus) and
# `source_decoy_protocol` (every decoy links to its protocol) because they
# would saturate the path atlas with "leak by being from the same dataset".
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


def _build_leak_subgraph(kg_dir: Path) -> pl.DataFrame:
    """Edges we walk when computing leak paths, with edge_type retained."""
    _, edges = load_canonical_kg(kg_dir)
    return edges.filter(pl.col("edge_type").is_in(list(_LEAK_EDGE_TYPES))).select(
        ["src", "dst", "edge_type"]
    )


def _bfs_path_typed(
    seeds: pl.DataFrame,
    targets: pl.DataFrame,
    edges: pl.DataFrame,
    max_hop: int,
) -> pl.DataFrame:
    """Multi-source BFS from `seeds` toward `targets`. For every target node
    that becomes reachable, return (target_node_id, hop, path_signature)
    where `path_signature` is the sorted-tuple of edge_types along the
    shortest path. We track the signature lazily (semi-canonical: each
    reached node remembers ONE representative incoming path).

    This implementation prioritises per-test reporting volume over
    completeness — when two paths of equal length exist, we record the one
    discovered first by the join order.
    """
    if not seeds.height or not targets.height:
        return pl.DataFrame(schema={"node_id": pl.Utf8, "hop": pl.Int64, "path_signature": pl.Utf8})
    # Symmetrise: keep edge_type on both directions.
    e_sym = pl.concat([
        edges.select(["src", "dst", "edge_type"]),
        edges.select([
            pl.col("dst").alias("src"),
            pl.col("src").alias("dst"),
            pl.col("edge_type"),
        ]),
    ], how="vertical_relaxed").unique()

    visited = seeds.select("node_id").unique().with_columns([
        pl.lit(0, dtype=pl.Int64).alias("hop"),
        pl.lit("", dtype=pl.Utf8).alias("path_signature"),
    ])
    frontier = visited
    target_set = targets.select("node_id").unique()
    hits: list[pl.DataFrame] = []
    for h in range(1, max_hop + 1):
        nbrs = (
            e_sym.join(frontier.select(pl.col("node_id").alias("src"), pl.col("path_signature")),
                       on="src", how="inner")
                 .select([
                     pl.col("dst").alias("node_id"),
                     pl.col("edge_type"),
                     pl.col("path_signature"),
                 ])
        )
        # Drop already-visited; collapse to one row per node (first-seen).
        new = nbrs.join(visited.select("node_id"), on="node_id", how="anti")
        if not new.height:
            break
        new = (new.with_columns(
                pl.when(pl.col("path_signature") == "")
                  .then(pl.col("edge_type"))
                  .otherwise(pl.col("path_signature") + pl.lit("|") + pl.col("edge_type"))
                  .alias("path_signature"))
                  .group_by("node_id")
                  .agg([pl.col("path_signature").first()])
                  .with_columns(pl.lit(h, dtype=pl.Int64).alias("hop")))
        # Record hits that are in target set.
        h_targets = new.join(target_set, on="node_id", how="semi")
        if h_targets.height:
            hits.append(h_targets.select(["node_id", "hop", "path_signature"]))
        visited = pl.concat([visited, new.select(["node_id", "hop", "path_signature"])],
                            how="vertical_relaxed")
        frontier = new
    if not hits:
        return pl.DataFrame(schema={"node_id": pl.Utf8, "hop": pl.Int64, "path_signature": pl.Utf8})
    return pl.concat(hits, how="vertical_relaxed").unique(subset=["node_id"], keep="first")


def _classify_signature(sig: str) -> str:
    """Reduce a `|`-joined edge-type sequence to a coarse leak category.

    Rules (priority order):
      - exact / fingerprint / parent ligand match → "exact_ligand"
      - similarity Tanimoto edge → "similar_ligand"
      - same scaffold → "scaffold"
      - same publication → "publication"
      - same assay → "assay"
      - protein cluster → "protein"
      - otherwise → first edge type (verbatim)
    """
    if not sig:
        return "(seed)"
    parts = sig.split("|")
    if any(p in {"ligand_exact", "ligand_parent_exact", "ligand_fingerprint_exact"} for p in parts):
        return "exact_ligand"
    if "ligand_similar" in parts:
        return "similar_ligand"
    if "ligand_scaffold" in parts:
        return "scaffold"
    if "example_from_publication" in parts:
        return "publication"
    if "example_from_assay" in parts:
        return "assay"
    if "protein_in_cluster" in parts:
        return "protein_cluster"
    return parts[0]


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema = PredictionSchema()
    preds = load_predictions(args.predictions, schema)
    train_df, test_df = split_train_test(preds, schema)

    # Identify mispredicts (default: positive whose score < threshold OR
    # negative whose score > threshold).
    th = args.mispredict_threshold
    mis = test_df.filter(
        ((pl.col(schema.label) == 1) & (pl.col(schema.score) < th))
        | ((pl.col(schema.label) == 0) & (pl.col(schema.score) >= th))
    )
    print(f"mispredict count at threshold={th}: {mis.height} / {test_df.height}")
    if not mis.height:
        (out / "report.md").write_text(
            "# Mảng B\nNo mispredicts at threshold; nothing to walk.\n", encoding="utf-8"
        )
        return

    edges = _build_leak_subgraph(Path(args.kg_dir))
    train_seeds = train_df.select(pl.col(schema.example_id).alias("node_id")).unique()
    mis_targets = mis.select(pl.col(schema.example_id).alias("node_id")).unique()

    paths = _bfs_path_typed(train_seeds, mis_targets, edges, max_hop=args.max_hop)
    paths = paths.with_columns(
        pl.col("path_signature").map_elements(_classify_signature, return_dtype=pl.Utf8).alias("category")
    )

    # Join back to mispredict rows, then to source.
    examples = load_examples(Path(args.kg_dir)).select(["node_id", "source"])
    per_mispredict = (
        mis.rename({schema.example_id: "node_id"})
        .join(paths, on="node_id", how="left")
        .join(examples, on="node_id", how="left")
        .with_columns([
            pl.col("category").fill_null("unreachable"),
            pl.col("hop").fill_null(-1),
        ])
    )
    per_mispredict.write_parquet(out / "per_mispredict_path.parquet")

    counts = (per_mispredict.group_by(["source", "category"]).len()
              .sort(["source", "len"], descending=[False, True])
              .rename({"len": "n_mispredict"}))
    counts.write_csv(out / "path_signature_counts.csv")

    top = (counts.group_by("source")
           .agg([pl.col("category"), pl.col("n_mispredict")])
           .with_columns([pl.col("category").list.head(3).alias("top_categories"),
                          pl.col("n_mispredict").list.head(3).alias("top_counts")]))
    lines = ["# Mảng B — Path-typed mispredict atlas", ""]
    lines.append(f"- mispredict total: {mis.height:,}")
    reachable = int(per_mispredict.filter(pl.col("hop") >= 0).height)
    lines.append(f"- reached via ≤{args.max_hop}-hop KG path to train: {reachable:,} "
                 f"({100*reachable/mis.height:.1f}%)")
    lines.append("")
    lines.append("## Top leak categories per corpus")
    lines.append(top.select(["source", "top_categories", "top_counts"]).to_pandas().to_markdown(index=False))
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}/per_mispredict_path.parquet")
    print(f"wrote {out}/path_signature_counts.csv")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--kg-dir", default=DEFAULT_KG_DIR, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-hop", type=int, default=4)
    p.add_argument("--mispredict-threshold", type=float, default=0.5,
                   help="score above which the model says 'active'")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
