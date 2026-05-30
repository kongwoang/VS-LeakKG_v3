"""Hydrate a split parquet into (example_id, fold, label, smiles) rows by
joining the canonical KG.

Split parquet schema (from `mang_C_kg_disjoint_split` outputs):
    node_id : str   — Example node_id from canonical KG
    fold    : str   — "train" | "test" | "val"

Output schema:
    example_id, fold, label, smiles, lig_node_id

When an Example has multiple ligands via the KG (rare, but possible from
wire), the FIRST one in join order is kept.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from ..common import (
    DEFAULT_KG_DIR,
    edges_of_types,
    load_canonical_kg,
    load_examples,
)


def hydrate_split(split: pl.DataFrame, kg_dir: Path | str = DEFAULT_KG_DIR) -> pl.DataFrame:
    examples = (
        load_examples(kg_dir)
        .select(["node_id", "label"])
        .rename({"node_id": "example_id"})
    )
    ex_lig = (
        edges_of_types(["example_has_ligand"], kg_dir)
        .select(["src", "dst"])
        .rename({"src": "example_id", "dst": "lig_node_id"})
        .group_by("example_id")
        .agg(pl.col("lig_node_id").first())
    )
    nodes, _ = load_canonical_kg(kg_dir)
    lig_smiles = (
        nodes.filter(pl.col("node_type") == "Ligand")
        .select(["node_id", "label"])
        .rename({"node_id": "lig_node_id", "label": "smiles"})
    )
    return (
        split.rename({"node_id": "example_id"})
        .join(examples, on="example_id", how="inner")
        .join(ex_lig, on="example_id", how="left")
        .join(lig_smiles, on="lig_node_id", how="left")
        .filter(pl.col("smiles").is_not_null() & (pl.col("smiles") != ""))
    )
