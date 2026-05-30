"""BFS and multi-hop neighbour queries on the canonical KG.

All routines are polars-native (no networkx) so they scale to the 38M-edge
graph without leaving Arrow. Edges are treated as undirected unless the
caller pre-orients them.

Two main primitives:

  bfs_distance(seeds, edges, max_hop)
      Returns (node_id, hop) for every node reachable within max_hop from
      the seed set. Hop 0 == the seed itself.

  distance_to_anchor_set(query_ids, anchor_ids, edges, max_hop)
      Returns (query_id, hop) where hop is the minimum BFS distance from
      query_id to any anchor (≥0) or null when unreachable within max_hop.
      Used to answer "how close is each test example to its nearest train?".
"""
from __future__ import annotations

import polars as pl


def _symmetric(edges: pl.DataFrame) -> pl.DataFrame:
    """Build undirected adjacency from a (src, dst, ...) frame."""
    return pl.concat([
        edges.select(["src", "dst"]),
        edges.select([pl.col("dst").alias("src"), pl.col("src").alias("dst")]),
    ], how="vertical_relaxed").unique()


def bfs_distance(
    seeds: pl.DataFrame,
    edges: pl.DataFrame,
    max_hop: int = 5,
    *,
    already_symmetric: bool = False,
) -> pl.DataFrame:
    """Multi-source BFS up to `max_hop` hops.

    Parameters
    ----------
    seeds : DataFrame with a `node_id` column. Treated as hop-0.
    edges : DataFrame with `src` and `dst` columns.
    max_hop : stop after this many expansions.
    already_symmetric : skip the (src↔dst) duplication if you've pre-done it.

    Returns
    -------
    DataFrame (node_id, hop) where hop ∈ [0, max_hop].
    """
    if not seeds.height:
        return pl.DataFrame(schema={"node_id": pl.Utf8, "hop": pl.Int64})
    e = edges if already_symmetric else _symmetric(edges)
    visited = seeds.select("node_id").unique().with_columns(
        pl.lit(0, dtype=pl.Int64).alias("hop")
    )
    frontier = visited
    for h in range(1, max_hop + 1):
        # Out-neighbours of the current frontier.
        nbrs = (
            e.join(frontier.select(pl.col("node_id").alias("src")), on="src", how="semi")
             .select(pl.col("dst").alias("node_id"))
             .unique()
        )
        new = nbrs.join(visited.select("node_id"), on="node_id", how="anti")
        if not new.height:
            break
        new = new.with_columns(pl.lit(h, dtype=pl.Int64).alias("hop"))
        visited = pl.concat([visited, new], how="vertical_relaxed")
        frontier = new
    return visited


def k_hop_neighbors(
    seeds: pl.DataFrame, edges: pl.DataFrame, k: int = 2
) -> pl.DataFrame:
    """Convenience: nodes strictly within k hops of seeds (exclude seeds themselves)."""
    out = bfs_distance(seeds, edges, max_hop=k)
    return out.filter((pl.col("hop") > 0) & (pl.col("hop") <= k))


def distance_to_anchor_set(
    query_ids: pl.DataFrame,
    anchor_ids: pl.DataFrame,
    edges: pl.DataFrame,
    max_hop: int = 5,
) -> pl.DataFrame:
    """For each query, return min-hop to any anchor (or null if > max_hop).

    Implementation: BFS from the anchor set, then left-join on query_ids.

    Parameters
    ----------
    query_ids : (node_id,) — the items to measure (e.g., test examples).
    anchor_ids : (node_id,) — the items to measure against (e.g., train examples).
    edges : (src, dst, ...) — the leak graph.
    max_hop : BFS cap.

    Returns
    -------
    DataFrame (node_id, hop) for every query_id, hop ∈ [0, max_hop] ∪ {null}.
    """
    reached = bfs_distance(
        anchor_ids.select("node_id").rename({"node_id": "node_id"}),
        edges,
        max_hop=max_hop,
    )
    return (
        query_ids.select("node_id")
        .unique()
        .join(reached, on="node_id", how="left")
    )


def axis_edges(kg_edges: pl.DataFrame, axis: str) -> pl.DataFrame:
    """Filter edges to those belonging to a given axis (per schema.AXIS_EDGE_TYPES).

    `axis` ∈ {ligand, scaffold, protein, assay, source, time}.
    """
    from vsleakkg.kg.schema import AXIS_EDGE_TYPES  # local import to avoid cycle

    if axis not in AXIS_EDGE_TYPES:
        raise KeyError(f"unknown axis '{axis}' (choose from {sorted(AXIS_EDGE_TYPES)})")
    types = AXIS_EDGE_TYPES[axis]
    return kg_edges.filter(pl.col("edge_type").is_in(types))
