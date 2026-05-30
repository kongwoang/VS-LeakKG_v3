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
    """Predict each test's score as the mean label of train items within
    k_hop on the leak subgraph.

    Implementation notes:
      - We don't materialise a symmetrised 2x edges frame — that blew up
        OOM on 38M edges. Instead, we look up neighbours via forward and
        reverse joins separately and concat.
      - For k_hop ≥ 2 we expand via a per-hop frontier (similar to
        bfs_distance) but propagate train labels along, capping each test
        to its nearest reached train hop so the average isn't dominated
        by distant noisy labels.
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
    train_ids = (enriched.filter(pl.col("fold") == "train")
                 .select(pl.col("example_id"),
                         pl.col("label").alias("train_label")))
    if not train_ids.height:
        raise SystemExit("split has 0 train rows")
    test_ids = enriched.filter(pl.col("fold") == "test").select("example_id")
    print(f"train={train_ids.height:,}, test={test_ids.height:,}", flush=True)

    # ---- k=1: direct neighbours via fwd + bwd joins (no symmetrise) ----
    # forward: leak.src ∈ test, leak.dst ∈ train
    fwd = (leak.join(test_ids.rename({"example_id": "src"}), on="src", how="semi")
               .join(train_ids.rename({"example_id": "dst",
                                       "train_label": "tl"}),
                     on="dst", how="inner")
               .select([pl.col("src").alias("test_id"), pl.col("tl")]))
    # backward: leak.dst ∈ test, leak.src ∈ train
    bwd = (leak.join(test_ids.rename({"example_id": "dst"}), on="dst", how="semi")
               .join(train_ids.rename({"example_id": "src",
                                       "train_label": "tl"}),
                     on="src", how="inner")
               .select([pl.col("dst").alias("test_id"), pl.col("tl")]))
    one_hop = pl.concat([fwd, bwd], how="vertical_relaxed").unique()
    print(f"  1-hop test→train edges: {one_hop.height:,}", flush=True)

    score_parts = [one_hop.with_columns(pl.lit(1, dtype=pl.Int64).alias("hop"))]
    if k_hop >= 2:
        # 2-hop: (test_id, mid) where mid is some intermediate non-train node.
        inter_fwd = (leak.join(test_ids.rename({"example_id": "src"}),
                               on="src", how="semi")
                         .select([pl.col("src").alias("test_id"),
                                  pl.col("dst").alias("mid")]))
        inter_bwd = (leak.join(test_ids.rename({"example_id": "dst"}),
                               on="dst", how="semi")
                         .select([pl.col("dst").alias("test_id"),
                                  pl.col("src").alias("mid")]))
        inter = pl.concat([inter_fwd, inter_bwd], how="vertical_relaxed").unique()
        # Drop intermediates that are themselves train (those edges already
        # counted as 1-hop).
        inter = inter.join(train_ids.select(pl.col("example_id").alias("mid")),
                           on="mid", how="anti")
        print(f"  2-hop intermediates: {inter.height:,}", flush=True)

        mid_set = inter.select("mid").unique()
        # mid -> train via either direction.
        mid_to_train_fwd = (leak.join(mid_set.rename({"mid": "src"}),
                                       on="src", how="semi")
                                .join(train_ids.rename({"example_id": "dst",
                                                        "train_label": "tl"}),
                                      on="dst", how="inner")
                                .select([pl.col("src").alias("mid"),
                                         pl.col("tl")]))
        mid_to_train_bwd = (leak.join(mid_set.rename({"mid": "dst"}),
                                       on="dst", how="semi")
                                .join(train_ids.rename({"example_id": "src",
                                                        "train_label": "tl"}),
                                      on="src", how="inner")
                                .select([pl.col("dst").alias("mid"),
                                         pl.col("tl")]))
        mid_to_train = pl.concat([mid_to_train_fwd, mid_to_train_bwd],
                                 how="vertical_relaxed").unique()
        two_hop = (inter.join(mid_to_train, on="mid", how="inner")
                        .select(["test_id", "tl"])
                        .with_columns(pl.lit(2, dtype=pl.Int64).alias("hop")))
        print(f"  2-hop test→train edges: {two_hop.height:,}", flush=True)
        score_parts.append(two_hop)

    all_hops = pl.concat(score_parts, how="vertical_relaxed")
    # For each test, take the AVERAGE label only over its nearest-hop
    # train neighbours (avoids distant noise overwhelming close signal).
    nearest = (all_hops.group_by("test_id").agg(pl.col("hop").min().alias("min_hop")))
    nearest_score = (
        all_hops.join(nearest, on="test_id", how="inner")
                .filter(pl.col("hop") == pl.col("min_hop"))
                .group_by("test_id")
                .agg(pl.col("tl").mean().alias("score"))
                .rename({"test_id": "example_id"})
    )

    # Train rows get score = own label (perfect by construction).
    train_self_score = train_ids.select([
        pl.col("example_id"), pl.col("train_label").alias("score")
    ])
    full_score = (
        enriched.join(
            pl.concat([train_self_score, nearest_score],
                      how="vertical_relaxed"),
            on="example_id", how="left",
        )
        .with_columns(pl.col("score").cast(pl.Float64).fill_null(default_score))
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
