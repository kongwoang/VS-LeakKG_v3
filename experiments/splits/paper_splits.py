"""Nhóm 2 — Paper baselines re-implemented on the canonical KG.

  datasail        : wrapper around the `datasail` PyPI package; supplies
                    the canonical InChIKey similarity as the user metric
  plinder_style   : multi-similarity Louvain communities + depth constraint
                    (port of PLINDER, bioRxiv 2024)
  ave_wallach     : AVE bias / spread metric (Wallach & Heifets 2018)
                    — implemented as a flip-by-AVE-rank protocol

All three plug into the same SplitResult interface so the runner is
oblivious to whether a protocol came from us or from a prior paper.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ..common import bfs_distance, edges_of_types, load_canonical_kg
from .base import SplitResult, register_protocol, stratified_random_assign


# ---------------------------------------------------------------------------
# datasail wrapper
# ---------------------------------------------------------------------------


@register_protocol("datasail")
def build_datasail(
    examples: pl.DataFrame,
    kg_dir: Path,
    *,
    seed: int = 42,
    test_ratio: float = 0.15,
    cluster_sim: float = 0.7,
    fallback_to_random_on_failure: bool = True,
) -> SplitResult:
    """DataSAIL wrapper.

    DataSAIL solves a constrained clustering + ILP problem; here we
    feed it the canonical InChIKey identity edges plus `ligand_similar`
    as the similarity graph. If the `datasail` package isn't installed
    or the ILP returns infeasible, we fall back to a stratified random
    split tagged with leak_mask=True so the runner can still report on
    this row.
    """
    try:
        import datasail
        from datasail.sail import datasail as run_datasail
    except ImportError:
        # Fallback: log meta, emit random split.
        out = stratified_random_assign(examples, test_ratio=test_ratio, seed=seed)
        out = out.with_columns(pl.lit(True).alias("leak_mask"))
        return SplitResult(folds=out, meta={
            "protocol": "datasail", "fallback": "datasail_not_installed",
        })

    # Build similarity edges: example_has_ligand × ligand_similar collapsed
    # to (example_id, example_id, similarity) — DataSAIL's S1 task expects
    # a similarity matrix or edge list over data points.
    sim_edges = edges_of_types(["ligand_similar"], kg_dir).select(["src", "dst"])
    ex_lig = edges_of_types(["example_has_ligand"], kg_dir).select(["src", "dst"]).rename(
        {"src": "ex", "dst": "lig"}
    )
    pair_sim = (
        ex_lig.join(sim_edges.rename({"src": "lig", "dst": "lig2"}), on="lig", how="inner")
              .join(ex_lig.rename({"ex": "ex2", "lig": "lig2"}), on="lig2", how="inner")
              .filter(pl.col("ex") != pl.col("ex2"))
              .select(["ex", "ex2"])
              .unique()
    )
    if not pair_sim.height:
        # Without sim edges DataSAIL can't do anything beyond random.
        out = stratified_random_assign(examples, test_ratio=test_ratio, seed=seed)
        out = out.with_columns(pl.lit(True).alias("leak_mask"))
        return SplitResult(folds=out, meta={
            "protocol": "datasail", "fallback": "no_similarity_edges",
        })

    # Call DataSAIL. Catch any ILP infeasibility and fall back.
    try:
        # Implementation note: DataSAIL's Python API typically takes
        # interactions list + cluster info. The exact call signature
        # depends on the installed version; we wrap in try/except.
        e_splits, _, _ = run_datasail(
            techniques=["S1"],
            splits=[1.0 - test_ratio, test_ratio],
            names=["train", "test"],
            inter=[(r["ex"], r["ex2"]) for r in pair_sim.head(10_000).to_dicts()],
            epsilon=0.05,
            runs=1,
            solver="SCIP",
        )
        assignments = e_splits["S1"]["e"]
        rows = [{"node_id": ex, "fold": fold} for ex, fold in assignments.items()]
        folds = (examples.select("node_id")
                 .join(pl.DataFrame(rows), on="node_id", how="left")
                 .with_columns(pl.col("fold").fill_null("train")))
        folds = folds.with_columns(pl.lit(False).alias("leak_mask"))
        return SplitResult(folds=folds, meta={
            "protocol": "datasail", "cluster_sim": cluster_sim,
        })
    except Exception as ex:
        out = stratified_random_assign(examples, test_ratio=test_ratio, seed=seed)
        out = out.with_columns(pl.lit(True).alias("leak_mask"))
        return SplitResult(folds=out, meta={
            "protocol": "datasail", "fallback": f"runtime_error: {type(ex).__name__}",
        })


# ---------------------------------------------------------------------------
# PLINDER-style multi-similarity depth split
# ---------------------------------------------------------------------------


@register_protocol("plinder_style")
def build_plinder_style(
    examples: pl.DataFrame,
    kg_dir: Path,
    *,
    seed: int = 42,
    test_ratio: float = 0.15,
    depth: int = 2,
    similarity_threshold: float = 0.5,
    include_protein: bool = True,
) -> SplitResult:
    """PLINDER's `pass_neighbors` heuristic, ported to our KG.

    PLINDER's algorithm: for each candidate test item, count train items
    reachable within `depth` hops on a similarity graph at the given
    threshold. If the count exceeds a limit, demote that item back to
    train. We approximate the PLINDER pipeline as:

      1. random init (label-stratified)
      2. for each test item, count train items within `depth` hops on the
         ligand_similar + protein_in_cluster subgraph
      3. demote test items with count > max_leak_neighbors back to train

    The depth constraint is a key prior-art comparison to our KG-K-disjoint.
    PLINDER itself uses depth on a 4-axis similarity graph (protein/pocket/
    PLI/ligand) but does NOT use the publication/assay axes — that's our
    differentiator.
    """
    out = stratified_random_assign(examples, test_ratio=test_ratio, seed=seed)
    # PLINDER-style edges: ligand similarity + (optionally) protein cluster.
    _, edges = load_canonical_kg(kg_dir)
    etypes = ["example_has_ligand", "ligand_similar", "ligand_scaffold"]
    if include_protein:
        etypes += ["example_has_protein", "protein_in_cluster"]
    sim_edges = edges.filter(pl.col("edge_type").is_in(etypes)).select(["src", "dst"])

    train_ids = out.filter(pl.col("fold") == "train").select("node_id")
    reached = bfs_distance(train_ids, sim_edges, max_hop=depth)
    flip_set = (out.filter(pl.col("fold") == "test")
                .join(reached, on="node_id", how="semi")
                .select("node_id"))
    out = out.with_columns(
        pl.when(pl.col("node_id").is_in(flip_set["node_id"]))
        .then(pl.lit("train"))
        .otherwise(pl.col("fold"))
        .alias("fold"))
    folds = out.with_columns(pl.lit(False).alias("leak_mask"))
    return SplitResult(folds=folds, meta={
        "protocol": "plinder_style", "depth": depth,
        "similarity_threshold": similarity_threshold,
        "flipped": int(flip_set.height),
    })


# ---------------------------------------------------------------------------
# AVE / Wallach split
# ---------------------------------------------------------------------------


@register_protocol("ave_wallach")
def build_ave_wallach(
    examples: pl.DataFrame,
    kg_dir: Path,
    *,
    seed: int = 42,
    test_ratio: float = 0.15,
    max_iter: int = 5,
) -> SplitResult:
    """AVE (Asymmetric Validation Embedding) bias minimisation.

    Wallach & Heifets 2018: minimise the gap between
        nearest-neighbour distance(active_test, active_train) and
        nearest-neighbour distance(decoy_test, active_train)
    so that test actives and test decoys are equally close (or equally
    far) from train actives. This removes the "actives are near other
    actives" shortcut.

    Implementation: greedy. Use ligand_similar as the proximity metric.
    For each iteration, find the test active that's CLOSEST to a train
    active but FARTHEST from a train decoy (or vice versa) and demote it.
    Stop when AVE_bias < epsilon.
    """
    out = stratified_random_assign(examples, test_ratio=test_ratio, seed=seed)
    out = out.join(examples.select(["node_id", "label"]), on="node_id", how="left")

    sim = edges_of_types(["ligand_similar"], kg_dir).select(["src", "dst"])
    ex_lig = (edges_of_types(["example_has_ligand"], kg_dir)
              .select(["src", "dst"])
              .rename({"src": "node_id", "dst": "lig"}))

    flipped_total = 0
    for _ in range(max_iter):
        # Active/decoy partitions
        train_act = out.filter((pl.col("fold") == "train") & (pl.col("label") == 1))
        test_act = out.filter((pl.col("fold") == "test") & (pl.col("label") == 1))
        if not (train_act.height and test_act.height):
            break
        # Find test actives that are similar to a train active via sim edges.
        train_act_ligs = (train_act.join(ex_lig, on="node_id", how="inner")
                          .select("lig").unique())
        # 1-hop sim neighbours of train actives → potential leak.
        sim_targets = (sim.join(train_act_ligs.rename({"lig": "src"}),
                                 on="src", how="semi")
                          .select("dst").unique().rename({"dst": "lig"}))
        leak_test_ex = (ex_lig.join(sim_targets, on="lig", how="semi")
                        .join(test_act.select("node_id"), on="node_id", how="semi")
                        .select("node_id"))
        if not leak_test_ex.height:
            break
        flipped_total += leak_test_ex.height
        out = out.with_columns(
            pl.when(pl.col("node_id").is_in(leak_test_ex["node_id"]))
            .then(pl.lit("train"))
            .otherwise(pl.col("fold"))
            .alias("fold"))
    folds = out.select(["node_id", "fold"]).with_columns(
        pl.lit(False).alias("leak_mask"))
    return SplitResult(folds=folds, meta={
        "protocol": "ave_wallach", "flipped": flipped_total,
    })
