"""C-NN — Contamination Nearest-Neighbour baseline.

For each test example, predict its score as the mean label of the train
examples within K hops on the KG leak subgraph. This baseline uses *only*
KG structure (no ligand structure, no protein sequence), so its AUROC is a
direct measure of how much label signal is encoded in the KG topology —
i.e. how much "free win" a model can get from contamination.

If C-NN approaches Morgan-RF performance on a benchmark, the benchmark's
predictability is mostly an artefact of train/test entanglement on the KG.

CLI
---
python -m experiments.baselines.cnn \
    --split outputs/experiments/mang_C/split_random__DEKOIS.parquet \
    --output predictions/cnn__random__DEKOIS.parquet \
    --k-hop 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from ..common import (
    DEFAULT_KG_DIR,
    bfs_distance,
    load_canonical_kg,
    load_examples,
)


# C-NN walks only the cross-example axes (ligand / scaffold / protein /
# publication / assay). example_has_ligand and example_has_protein are
# included so the walk passes through these intermediate nodes; the score
# of each test then reflects the mean label over train examples reachable
# through ANY of these axes.
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


def cnn_score(
    split: pl.DataFrame, kg_dir: Path, k_hop: int = 2, default_score: float = 0.5
) -> pl.DataFrame:
    """Return predictions with `score = mean(train_label within k hops)`.

    Items that have no train neighbour within k_hop receive `default_score`
    (the corpus prior). Per-row schema matches PredictionSchema.
    """
    _, edges = load_canonical_kg(kg_dir)
    leak = edges.filter(pl.col("edge_type").is_in(list(_LEAK_EDGE_TYPES))).select(
        ["src", "dst"]
    )
    examples = load_examples(kg_dir).select(["node_id", "label"]).rename(
        {"node_id": "example_id"}
    )
    enriched = split.rename({"node_id": "example_id"}).join(
        examples, on="example_id", how="inner"
    )
    train_ids = enriched.filter(pl.col("fold") == "train").select(
        pl.col("example_id").alias("node_id"), pl.col("label").alias("train_label")
    )
    if not train_ids.height:
        raise SystemExit("split has 0 train rows")
    # BFS from every train seed; carry their label along so we can average.
    reached = bfs_distance(
        train_ids.select("node_id"), leak, max_hop=k_hop
    ).rename({"hop": "min_hop"})
    # For each reached node, collect labels of the train items it could have
    # been reached from. This requires a per-train BFS in the general case;
    # we approximate it by averaging labels of train items reachable from
    # the *reached* node (BFS is symmetric on this undirected subgraph).
    reached_with_label = reached.join(train_ids, on="node_id", how="left")
    # The reverse: for each test node, find the train neighbours via the
    # symmetric edge set.
    # Build (test_node_id, [train_label]) by running BFS rooted at train
    # and grouping by which train was the seed — too expensive to do
    # per-seed for 5M train items, so use the heuristic that "if the
    # reached node is itself labelled in train, use its label; otherwise
    # join the symmetric edge once and pull adjacent train labels".
    e_sym = pl.concat([
        leak.select(["src", "dst"]),
        leak.select([pl.col("dst").alias("src"), pl.col("src").alias("dst")]),
    ], how="vertical_relaxed").unique()
    test_ids = enriched.filter(pl.col("fold") == "test").select("example_id")
    # For each test, look at all direct neighbours that are labelled in
    # train, and take the mean. This is a 1-hop variant of C-NN. For 2-hop,
    # repeat through one more join.
    train_label_map = train_ids
    test_to_nbr = (
        e_sym.join(test_ids.rename({"example_id": "src"}), on="src", how="semi")
             .select([pl.col("src").alias("example_id"), pl.col("dst").alias("nbr")])
    )
    if k_hop >= 2:
        # second hop: nbr -> 2-hop neighbours
        nbr2 = (
            e_sym.join(test_to_nbr.select(pl.col("nbr").alias("src")),
                       on="src", how="semi")
                 .select([pl.col("src").alias("nbr"), pl.col("dst").alias("nbr2")])
        )
        test_to_2hop = test_to_nbr.join(nbr2, on="nbr", how="inner").select([
            "example_id", pl.col("nbr2").alias("nbr"),
        ])
        test_to_nbr = pl.concat([test_to_nbr, test_to_2hop],
                                how="vertical_relaxed").unique()
    # Lookup train labels.
    train_label_lookup = train_label_map.rename({"node_id": "nbr"})
    test_nbr_label = test_to_nbr.join(train_label_lookup, on="nbr", how="inner")
    test_score = (
        test_nbr_label.group_by("example_id")
        .agg(pl.col("train_label").mean().alias("score"))
    )
    # Train rows get score = own label (perfect by construction). Test rows
    # get the computed score; unreached test items get default_score.
    train_self_score = train_ids.rename({"node_id": "example_id", "train_label": "score"})
    full_score = (
        enriched.join(
            pl.concat([train_self_score, test_score], how="vertical_relaxed"),
            on="example_id", how="left",
        )
        .with_columns(pl.col("score").fill_null(default_score))
    )
    return full_score.select(["example_id", "score", "label", "fold"])


def run(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    split = pl.read_parquet(args.split)
    print(f"loaded split: {split.height} rows", flush=True)
    scored = cnn_score(split, Path(args.kg_dir), k_hop=args.k_hop,
                       default_score=args.default_score)
    out_df = scored.with_columns(pl.lit(args.model_name).alias("model"))
    out_df.write_parquet(out)
    print(f"wrote {out} ({out_df.height} rows)", flush=True)


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kg-dir", type=Path, default=DEFAULT_KG_DIR)
    p.add_argument("--k-hop", type=int, default=2)
    p.add_argument("--default-score", type=float, default=0.5)
    p.add_argument("--model-name", default="cnn")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
