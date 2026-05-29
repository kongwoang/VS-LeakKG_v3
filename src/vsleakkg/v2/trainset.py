"""Model-specific TrainSet ingestion (Mode B audits).

Each evaluated model gets a single TrainSet_m node and one
example_in_trainset_m edge per training row. Per proposal section 5.6:

    D_train^m = { x_i : x_i -> TrainSet_m }
    C_m(x_t)  = C(x_t, D_train^m)

This module accepts a model manifest (a parquet/CSV listing the training
identifiers) and produces:
  - one TrainSet_m node
  - example_in_trainset edges into the v2 graph

Manifest formats supported:
  - parquet with columns (model, example_id) or (model, source, identifier)
  - CSV with the same columns

If a manifest lists external IDs (e.g. BindingDB row IDs) that don't yet
correspond to Example nodes in the v2 graph, we attempt to resolve them via
an id_map DataFrame keyed by (source, identifier) -> example_id. Unresolved
rows are returned as a separate dataframe and not silently dropped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl

from .schema import EdgeType, NodeType


def _read_manifest(path: str | Path) -> pl.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pl.read_parquet(p)
    if p.suffix.lower() in (".csv", ".tsv"):
        sep = "," if p.suffix.lower() == ".csv" else "\t"
        return pl.read_csv(p, separator=sep)
    raise ValueError(f"unsupported manifest format: {p.suffix}")


def ingest_model_trainset(
    manifest_path: str | Path,
    *,
    model_id: str,
    id_map: pl.DataFrame | None = None,
) -> dict[str, pl.DataFrame]:
    """Convert a model training manifest into v2 graph nodes/edges.

    Args:
        manifest_path: parquet/CSV listing training rows. Must contain a
                       'model' column matching `model_id`, and either an
                       'example_id' column (preferred) or both 'source' and
                       'identifier' columns (which get resolved through
                       `id_map`).
        model_id:      identifier for the TrainSet node, e.g. 'conglude'.
        id_map:        DataFrame with (source, identifier, example_id) for
                       resolving non-VS-LeakKG row IDs.

    Returns:
        dict with keys:
          'nodes'      -> DataFrame with a single TrainSet_m node
          'edges'      -> DataFrame with one example_in_trainset edge per row
          'unresolved' -> DataFrame of manifest rows that couldn't be mapped
    """
    df = _read_manifest(manifest_path)
    if "model" in df.columns:
        df = df.filter(pl.col("model") == model_id)
    if df.height == 0:
        raise ValueError(f"manifest contains no rows for model {model_id!r}")

    if "example_id" in df.columns:
        mapped = df.select("example_id")
        unresolved = df.head(0)
    else:
        if id_map is None:
            raise ValueError(
                "manifest lacks example_id and no id_map was provided"
            )
        if not all(c in df.columns for c in ("source", "identifier")):
            raise ValueError(
                "manifest must have either example_id, or both source+identifier"
            )
        joined = df.join(id_map, on=("source", "identifier"), how="left")
        mapped = joined.filter(pl.col("example_id").is_not_null()).select("example_id")
        unresolved = joined.filter(pl.col("example_id").is_null()).drop("example_id")

    trainset_node_id = f"trainset:{model_id}"
    nodes = pl.DataFrame([{
        "node_id": trainset_node_id,
        "node_type": NodeType.TRAINSET.value,
        "label": model_id,
        "props": f'{{"model": "{model_id}"}}',
    }])

    edges = mapped.with_columns([
        pl.lit(trainset_node_id).alias("dst"),
        pl.lit(EdgeType.EXAMPLE_IN_TRAINSET.value).alias("edge_type"),
        pl.lit(f'{{"model": "{model_id}"}}').alias("props"),
    ]).rename({"example_id": "src"}).select(["src", "dst", "edge_type", "props"])

    return {"nodes": nodes, "edges": edges, "unresolved": unresolved}


def merge_into_graph(
    graph_nodes: pl.DataFrame,
    graph_edges: pl.DataFrame,
    trainset: dict[str, pl.DataFrame],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Append TrainSet nodes/edges to an existing v2 graph.

    Idempotent: if the TrainSet node is already present (same node_id), the
    existing entry is preserved and only new edges are appended. Duplicate
    edges (same src, dst, edge_type) are deduped.
    """
    nodes = pl.concat([graph_nodes, trainset["nodes"]], how="diagonal").unique(
        subset=["node_id"], keep="first"
    )
    edges = pl.concat([graph_edges, trainset["edges"]], how="diagonal").unique(
        subset=["src", "dst", "edge_type"], keep="first"
    )
    return nodes, edges


def ingest_many(
    manifests: Iterable[tuple[str, str | Path]],
    *,
    id_map: pl.DataFrame | None = None,
) -> dict[str, dict[str, pl.DataFrame]]:
    """Convenience: ingest several model manifests at once.

    Args:
        manifests: iterable of (model_id, manifest_path).
        id_map:    optional shared identifier map.

    Returns:
        dict[model_id -> {'nodes', 'edges', 'unresolved'}]
    """
    out: dict[str, dict[str, pl.DataFrame]] = {}
    for model_id, path in manifests:
        out[model_id] = ingest_model_trainset(path, model_id=model_id, id_map=id_map)
    return out
