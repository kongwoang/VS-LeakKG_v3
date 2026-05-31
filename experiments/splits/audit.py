"""Audit metrics for a saved split. Computes feasibility + per-axis residual
leak without touching any model.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from ..common import bfs_distance, edges_of_types, load_canonical_kg
from .kg_splits import _AXIS_EDGE_SETS


# Feasibility: split is usable for AUROC if
#   (a) total test ≥ 500   (stable bootstrap CI)
#   (b) ≥ 10 actives       (need positives — works for LIT-PCBA's 0.3% rate)
#   (c) ≥ 10 decoys        (need negatives)
# Absolute counts (not %) so corpora with low natural active rate
# (LIT-PCBA = 0.3%) aren't falsely flagged.
_FEASIBILITY_MIN_N_TEST = 500
_FEASIBILITY_MIN_N_ACTIVE = 10
_FEASIBILITY_MIN_N_DECOY = 10


def audit_split(
    split: pl.DataFrame,
    examples: pl.DataFrame,
    kg_dir: Path,
    *,
    k_hops_for_leak: int = 2,
) -> dict:
    """Compute audit metrics on a (node_id, fold) split.

    `examples` provides label and source columns; `kg_dir` provides edges
    for per-axis residual-leak computation. Returns a flat dict suitable
    for one row in the audit summary CSV.
    """
    enriched = split.join(examples, on="node_id", how="left")
    train = enriched.filter(pl.col("fold") == "train")
    test = enriched.filter(pl.col("fold") == "test")
    val = enriched.filter(pl.col("fold") == "val")

    n_test_active = int((test["label"] == 1).sum()) if test.height else 0
    n_test_decoy = int((test["label"] == 0).sum()) if test.height else 0
    row: dict = {
        "n_train": train.height, "n_val": val.height, "n_test": test.height,
        "n_test_active": n_test_active, "n_test_decoy": n_test_decoy,
        "pct_active_train": float((train["label"] == 1).mean()) if train.height else None,
        "pct_active_test": float((test["label"] == 1).mean()) if test.height else None,
    }

    # Per-axis residual leak in test
    _, edges = load_canonical_kg(kg_dir)
    train_ids = train.select("node_id")
    for axis, etypes in _AXIS_EDGE_SETS.items():
        if not train_ids.height or not test.height:
            row[f"pct_leak_{axis}"] = None
            continue
        sub = edges.filter(pl.col("edge_type").is_in(list(etypes))).select(["src", "dst"])
        if not sub.height:
            row[f"pct_leak_{axis}"] = 0.0
            continue
        reached = bfs_distance(train_ids, sub, max_hop=k_hops_for_leak)
        n_leak = test.select("node_id").join(reached, on="node_id", how="semi").height
        row[f"pct_leak_{axis}"] = 100.0 * n_leak / test.height

    # Target diversity in test (via example_has_protein)
    ex_prot = edges_of_types(["example_has_protein"], kg_dir).select(
        ["src", "dst"]).rename({"src": "node_id", "dst": "prot"})
    test_targets = (test.join(ex_prot, on="node_id", how="inner")
                    .select("prot").unique().height)
    row["n_unique_targets_test"] = test_targets

    # Feasibility: enough test items to compute AUROC stably + non-degenerate
    # label balance (need at least 10 positives and 10 negatives).
    row["feasible"] = (
        test.height >= _FEASIBILITY_MIN_N_TEST
        and n_test_active >= _FEASIBILITY_MIN_N_ACTIVE
        and n_test_decoy >= _FEASIBILITY_MIN_N_DECOY
    )
    return row
