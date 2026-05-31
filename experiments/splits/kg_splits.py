"""Nhóm 3 — KG-based split protocols proposed in this work.

  kg_kdisjoint    : train and test must be ≥ K hops apart on the leak subgraph
                    (K ∈ {1, 2, 3}; K=3 typically empties the test on tight
                    benchmarks — that itself is a finding)
  kg_maxmin       : multi-axis generalization of Tanimoto-MaxMin. Iteratively
                    prune the test item with shortest KG distance to train
                    until the minimum distance ≥ T.
  kg_axis_budget  : per-axis residual-leak budget. Reject test items violating
                    any axis budget; equivalent to soft K-disjoint with
                    different K per axis.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from ..common import bfs_distance, load_canonical_kg
from .base import SplitResult, register_protocol, stratified_random_assign


_AXIS_EDGE_SETS: dict[str, tuple[str, ...]] = {
    "ligand": (
        "example_has_ligand", "ligand_similar", "ligand_exact",
        "ligand_parent_exact", "ligand_fingerprint_exact",
    ),
    "scaffold": ("example_has_ligand", "ligand_scaffold"),
    "protein": ("example_has_protein", "protein_in_cluster"),
    "publication": ("example_from_publication",),
    "assay": ("example_from_assay",),
}

_ALL_LEAK_EDGES: tuple[str, ...] = tuple(set().union(*_AXIS_EDGE_SETS.values()))


def _load_leak_edges(kg_dir: Path, axes: tuple[str, ...] | None = None) -> pl.DataFrame:
    """Return the subgraph used for KG-distance measurement."""
    _, edges = load_canonical_kg(kg_dir)
    if axes is None:
        etypes = list(_ALL_LEAK_EDGES)
    else:
        s: set[str] = set()
        for ax in axes:
            s.update(_AXIS_EDGE_SETS[ax])
        etypes = list(s)
    return edges.filter(pl.col("edge_type").is_in(etypes)).select(["src", "dst"])


@register_protocol("kg_kdisjoint")
def build_kg_kdisjoint(
    examples: pl.DataFrame,
    kg_dir: Path,
    *,
    seed: int = 42,
    test_ratio: float = 0.15,
    K: int = 2,
    max_iter: int = 3,
) -> SplitResult:
    """Iterative: random init → BFS from train ≤K hops → flip reached test
    items to train. Stops when no more flips or max_iter reached.
    """
    out = stratified_random_assign(examples, test_ratio=test_ratio, seed=seed)
    leak = _load_leak_edges(kg_dir)
    flipped_total = 0
    for _ in range(max_iter):
        train_ids = out.filter(pl.col("fold") == "train").select("node_id")
        if not train_ids.height:
            break
        reached = bfs_distance(train_ids, leak, max_hop=K)
        flip_set = (out.filter(pl.col("fold") == "test")
                    .join(reached, on="node_id", how="semi").select("node_id"))
        if not flip_set.height:
            break
        flipped_total += flip_set.height
        out = out.with_columns(
            pl.when(pl.col("node_id").is_in(flip_set["node_id"]))
            .then(pl.lit("train"))
            .otherwise(pl.col("fold"))
            .alias("fold"))
    folds = out.with_columns(pl.lit(False).alias("leak_mask"))
    return SplitResult(folds=folds, meta={
        "protocol": "kg_kdisjoint", "K": K, "flipped": flipped_total,
    })


@register_protocol("kg_maxmin")
def build_kg_maxmin(
    examples: pl.DataFrame,
    kg_dir: Path,
    *,
    seed: int = 42,
    test_ratio: float = 0.15,
    T: int = 2,
    max_iter: int = 5,
) -> SplitResult:
    """Multi-axis MaxMin: random init → iteratively prune the closest
    train-test pairs until min KG distance ≥ T.

    When edge set = {ligand_similar} this reduces to Tanimoto-MaxMin;
    when edge set = all leak axes it generalises to multi-axis MaxMin —
    the central contribution.
    """
    out = stratified_random_assign(examples, test_ratio=test_ratio, seed=seed)
    leak = _load_leak_edges(kg_dir)
    flipped_total = 0
    for _ in range(max_iter):
        train_ids = out.filter(pl.col("fold") == "train").select("node_id")
        reached = bfs_distance(train_ids, leak, max_hop=T - 1)
        # Test items at distance < T (i.e. within T-1 hops) → flip back to train.
        flip_set = (out.filter(pl.col("fold") == "test")
                    .join(reached, on="node_id", how="semi").select("node_id"))
        if not flip_set.height:
            break
        flipped_total += flip_set.height
        out = out.with_columns(
            pl.when(pl.col("node_id").is_in(flip_set["node_id"]))
            .then(pl.lit("train"))
            .otherwise(pl.col("fold"))
            .alias("fold"))
    folds = out.with_columns(pl.lit(False).alias("leak_mask"))
    return SplitResult(folds=folds, meta={
        "protocol": "kg_maxmin", "T": T, "flipped": flipped_total,
    })


@register_protocol("kg_axis_budget")
def build_kg_axis_budget(
    examples: pl.DataFrame,
    kg_dir: Path,
    *,
    seed: int = 42,
    test_ratio: float = 0.15,
    K: int = 2,
    budget_ligand: float = 0.05,
    budget_scaffold: float = 0.05,
    budget_publication: float = 0.05,
    budget_assay: float = 0.05,
    budget_protein: float = 0.50,
) -> SplitResult:
    """Per-axis residual leak budgets.

    For each test item, check the K-hop reach from train on each axis
    subgraph independently. A test item is REMOVED to train if it violates
    *any* budget; budgets are evaluated globally (i.e. a budget of 5%
    means at most 5% of the surviving test may be reachable on that axis).

    Greedy approach: sort test items by composite distance (lowest = worst
    leak), flip from train until each axis residual is under budget.
    """
    out = stratified_random_assign(examples, test_ratio=test_ratio, seed=seed)
    budgets = {
        "ligand": budget_ligand, "scaffold": budget_scaffold,
        "publication": budget_publication, "assay": budget_assay,
        "protein": budget_protein,
    }
    flipped_total = 0
    # For each axis, compute per-test "leak distance" and flip items
    # ordered by how leaky they are.
    for axis, budget in budgets.items():
        leak = _load_leak_edges(kg_dir, axes=(axis,))
        if not leak.height:
            continue
        train_ids = out.filter(pl.col("fold") == "train").select("node_id")
        reached = bfs_distance(train_ids, leak, max_hop=K).rename({"hop": "axis_hop"})
        test_items = out.filter(pl.col("fold") == "test").select("node_id")
        if not test_items.height:
            continue
        leaky_test = test_items.join(reached, on="node_id", how="inner")
        n_test = test_items.height
        n_allowed_leak = int(budget * n_test)
        if leaky_test.height <= n_allowed_leak:
            continue
        # Flip the (leaky_test.height - n_allowed_leak) closest to train.
        flip_set = (leaky_test.sort("axis_hop").head(leaky_test.height - n_allowed_leak)
                    .select("node_id"))
        flipped_total += flip_set.height
        out = out.with_columns(
            pl.when(pl.col("node_id").is_in(flip_set["node_id"]))
            .then(pl.lit("train"))
            .otherwise(pl.col("fold"))
            .alias("fold"))
    folds = out.with_columns(pl.lit(False).alias("leak_mask"))
    return SplitResult(folds=folds, meta={
        "protocol": "kg_axis_budget", "K": K, "flipped": flipped_total,
        "budgets": budgets,
    })
