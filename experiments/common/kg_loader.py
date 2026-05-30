"""Canonical KG loader + axis-edge slicing helpers.

Loads `outputs/kg/canonical_{nodes,edges}.parquet` with eager polars frames
and exposes convenience views (examples-with-source, edge subsets by type,
symmetrized adjacency). All loads are LRU-cached so repeated experiment
runs in the same Python process amortise the read cost.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import polars as pl

DEFAULT_KG_DIR = Path("outputs/kg")


@lru_cache(maxsize=4)
def load_canonical_kg(kg_dir: str | Path = DEFAULT_KG_DIR) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read the canonical nodes + edges parquets eagerly.

    Returns
    -------
    (nodes, edges): polars DataFrames as written by `vsleakkg.kg.consolidate`.
    """
    kg_dir = Path(kg_dir)
    nodes = pl.read_parquet(kg_dir / "canonical_nodes.parquet")
    edges = pl.read_parquet(kg_dir / "canonical_edges.parquet")
    return nodes, edges


def load_examples(kg_dir: str | Path = DEFAULT_KG_DIR) -> pl.DataFrame:
    """Return Examples with parsed `source` and `label` columns.

    Schema: (node_id, source, label, props).
    """
    nodes, _ = load_canonical_kg(kg_dir)
    ex = nodes.filter(pl.col("node_type") == "Example")
    return ex.with_columns([
        pl.col("props").str.json_path_match("$.source").alias("source"),
        pl.col("props").str.json_path_match("$.label").cast(pl.Int64).alias("label"),
    ]).select(["node_id", "source", "label", "props"])


def edges_of_types(
    edge_types: Iterable[str], kg_dir: str | Path = DEFAULT_KG_DIR
) -> pl.DataFrame:
    """Filter canonical edges to the given types. Returns (src, dst, edge_type, props)."""
    _, edges = load_canonical_kg(kg_dir)
    types = list(edge_types)
    return edges.filter(pl.col("edge_type").is_in(types))


def symmetrize_edges(edges: pl.DataFrame) -> pl.DataFrame:
    """Return undirected (src, dst) by concatenating reverse pairs. Drops props/weights."""
    fwd = edges.select(["src", "dst"])
    rev = edges.select([pl.col("dst").alias("src"), pl.col("src").alias("dst")])
    return pl.concat([fwd, rev], how="vertical_relaxed").unique()


def examples_by_corpus(
    corpus: str | None = None, kg_dir: str | Path = DEFAULT_KG_DIR
) -> pl.DataFrame:
    """Convenience: examples filtered to a single corpus (or all if None)."""
    ex = load_examples(kg_dir)
    if corpus is None:
        return ex
    return ex.filter(pl.col("source") == corpus)
