"""Build the v2 contamination graph from the v1 processed parquets.

The v1 repo (sibling directory, path resolved via `vsleakkg.v2.datapaths`)
already produced per-corpus and combined node/edge parquets and the
multi-resolution MMseqs2 protein clusters for PDBBind. This module
**consolidates** those into the v2 schema (`vsleakkg.v2.schema`) rather
than rebuilding from raw archives:

  v1 mvp2_nodes / mvp2_edges
            +
  v1 per-corpus *_nodes / *_edges   (LIT-PCBA-AVE, DUD-E, DEKOIS, PDBBind, BayesBind)
            +
  v1 pdbbind_protein_clusters_{30,50,90}.parquet

  → outputs/v2/graph/v2_nodes.parquet
  → outputs/v2/graph/v2_edges.parquet
  → outputs/v2/graph/stats.csv

The mapping is intentionally lossy. v1 carries scaffolding nodes the v2
audit doesn't need (`ChEMBLActivity`, `BindingMeasurement`, `Split`,
`LabelType`, `AffinityType`, `DatabaseRelease`, `StructureFile`,
`Complex`, `PDBBindSubset`); they are dropped. The v1 edges that depend
on those nodes are also dropped. What survives is the seven-axis
contamination graph from proposal.tex Table 2.

Edges that need an encoder we don't have on this box (pocket similarity
via ESM-IF1, time bins) are documented as TODOs in `stats.csv` so the
audit report can flag them.

CLI:

    python -m vsleakkg.v2.build_graph \
        --output-dir /vol/.../VS-LeakKG_v2/outputs/v2/graph \
        [--corpus litpcba_ave|dude|dekois|pdbbind|bayesbind|all]   # default all
        [--limit 100000]    # for smoke-testing

The script is idempotent: it will overwrite the output parquets.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from .datapaths import processed_dir, require_data_root
from .schema import (
    AXIS_EDGE_TYPES,
    DEFAULT_WEIGHTS,
    EdgeType,
    HubMitigationConfig,
    NodeType,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# v1 -> v2 mapping tables
# ---------------------------------------------------------------------------

V1_TO_V2_NODE_TYPE: dict[str, str] = {
    "Example": NodeType.EXAMPLE.value,
    "Ligand": NodeType.LIGAND.value,
    "Scaffold": NodeType.SCAFFOLD.value,
    "Protein": NodeType.PROTEIN.value,
    "ProteinTarget": NodeType.PROTEIN.value,
    "Pocket": NodeType.POCKET.value,
    "ChEMBLAssay": NodeType.ASSAY.value,
    "ChEMBLDocument": NodeType.PUBLICATION.value,
    "DatasetSource": NodeType.DATASET_SOURCE.value,
    "DecoyProtocol": NodeType.DECOY_PROTOCOL.value,
}

# v1 node types that are absorbed elsewhere or are v1-specific scaffolding.
V1_DROPPED_NODES: frozenset[str] = frozenset({
    "ChEMBLActivity",        # absorbed into Example (label, label_type)
    "BindingMeasurement",    # absorbed into Example
    "Complex",               # PDBBind-specific intermediary
    "StructureFile",         # not a leakage axis
    "PDBBindSubset",         # v1 partitioning artefact
    "AffinityType",          # static lookup table
    "LabelType",             # static lookup table
    "DatabaseRelease",       # version metadata
    "Split",                 # v1 split labels; v2 emits partition assignments separately
})

V1_TO_V2_EDGE_TYPE: dict[str, str] = {
    "example_has_ligand": EdgeType.EXAMPLE_HAS_LIGAND.value,
    "example_targets_protein": EdgeType.EXAMPLE_HAS_PROTEIN.value,
    "example_from_source": EdgeType.EXAMPLE_FROM_SOURCE.value,
    "ligand_has_scaffold": EdgeType.LIGAND_SCAFFOLD.value,
    "ligand_similar_to_ligand": EdgeType.LIGAND_SIMILAR.value,
    "same_inchikey_as": EdgeType.LIGAND_EXACT.value,
    "example_uses_decoy_protocol": EdgeType.SOURCE_DECOY_PROTOCOL.value,
    # complex_has_pocket: maps to example_has_pocket only after we collapse
    # Complex -> Example. We expand that below in `_expand_pdbbind_pockets`.
}

V1_DROPPED_EDGES: frozenset[str] = frozenset({
    "example_in_split",
    "example_has_label_type",
    "binding_measurement_has_type",
    "complex_has_structure_file",
    "complex_in_subset",
    "complex_from_source",
    "complex_has_binding_measurement",
    "complex_has_protein",
    "complex_has_ligand",
    "complex_has_pocket",
})


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class BuildStats:
    n_nodes_in: int = 0
    n_edges_in: int = 0
    n_nodes_dropped: int = 0
    n_edges_dropped: int = 0
    n_nodes_out: int = 0
    n_edges_out: int = 0
    nodes_by_type: dict[str, int] = None     # type: ignore[assignment]
    edges_by_type: dict[str, int] = None     # type: ignore[assignment]
    n_protein_cluster_edges: dict[str, int] = None  # type: ignore[assignment]
    n_trivial_scaffolds_dropped: int = 0
    n_hub_nodes_sharded: int = 0
    deferred: list[str] = None    # type: ignore[assignment]

    def to_csv_rows(self) -> list[dict]:
        rows: list[dict] = [
            {"key": "n_nodes_in", "value": self.n_nodes_in},
            {"key": "n_edges_in", "value": self.n_edges_in},
            {"key": "n_nodes_dropped", "value": self.n_nodes_dropped},
            {"key": "n_edges_dropped", "value": self.n_edges_dropped},
            {"key": "n_nodes_out", "value": self.n_nodes_out},
            {"key": "n_edges_out", "value": self.n_edges_out},
            {"key": "n_trivial_scaffolds_dropped",
             "value": self.n_trivial_scaffolds_dropped},
            {"key": "n_hub_nodes_sharded", "value": self.n_hub_nodes_sharded},
        ]
        for t, n in sorted((self.nodes_by_type or {}).items()):
            rows.append({"key": f"nodes_by_type::{t}", "value": int(n)})
        for t, n in sorted((self.edges_by_type or {}).items()):
            rows.append({"key": f"edges_by_type::{t}", "value": int(n)})
        for res, n in sorted((self.n_protein_cluster_edges or {}).items()):
            rows.append({"key": f"protein_cluster_edges::{res}", "value": int(n)})
        for d in self.deferred or []:
            rows.append({"key": f"deferred::{d}", "value": 0})
        return rows


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------


def _map_nodes(nodes: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Map v1 node_type to v2 node_type. Drop scaffolding-only v1 types."""
    keep = list(V1_TO_V2_NODE_TYPE.keys())
    n_in = nodes.height
    kept = nodes.filter(pl.col("node_type").is_in(keep)).with_columns(
        pl.col("node_type").replace(V1_TO_V2_NODE_TYPE).alias("node_type")
    )
    return kept, n_in - kept.height


def _map_edges(edges: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Map v1 edge_type to v2 edge_type. Drop scaffolding-only v1 edges."""
    keep = list(V1_TO_V2_EDGE_TYPE.keys())
    n_in = edges.height
    kept = edges.filter(pl.col("edge_type").is_in(keep)).with_columns(
        pl.col("edge_type").replace(V1_TO_V2_EDGE_TYPE).alias("edge_type")
    )
    return kept, n_in - kept.height


def _drop_trivial_scaffolds(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    cfg: HubMitigationConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Remove scaffold nodes with <= cfg.trivial_scaffold_max_atoms heavy atoms."""
    trivial_ids = (
        nodes.filter(pl.col("node_type") == NodeType.SCAFFOLD.value)
        .with_columns(
            pl.col("label")
            .str.replace_all(r"[^A-Za-z]", "")
            .str.len_chars()
            .alias("heavy_approx")
        )
        .filter(pl.col("heavy_approx") <= cfg.trivial_scaffold_max_atoms)
        ["node_id"]
        .to_list()
    )
    n_trivial = len(trivial_ids)
    if not trivial_ids:
        return nodes, edges, 0
    nodes_out = nodes.filter(~pl.col("node_id").is_in(trivial_ids))
    edges_out = edges.filter(
        ~pl.col("src").is_in(trivial_ids) & ~pl.col("dst").is_in(trivial_ids)
    )
    return nodes_out, edges_out, n_trivial


def _shard_hub_nodes(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    cfg: HubMitigationConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Apply degree cap: any node with degree > cfg.degree_cap gets is_hub=True."""
    deg_src = edges.group_by("src").agg(pl.len().alias("deg")).rename({"src": "node_id"})
    deg_dst = edges.group_by("dst").agg(pl.len().alias("deg")).rename({"dst": "node_id"})
    deg = (
        pl.concat([deg_src, deg_dst])
        .group_by("node_id")
        .agg(pl.col("deg").sum())
    )
    hubs = (
        deg.filter(pl.col("deg") > cfg.degree_cap)
        ["node_id"]
        .to_list()
    )
    n_hubs = len(hubs)
    nodes_out = nodes.with_columns(
        pl.col("node_id").is_in(hubs).alias("is_hub") if hubs
        else pl.lit(False).alias("is_hub")
    )
    return nodes_out, edges, n_hubs


def _synthesize_pdbbind_examples(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    raw_edges: pl.DataFrame,
    raw_nodes: pl.DataFrame,
    stats: BuildStats,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """For PDBBind: synthesize one Example node per Complex.

    The v1 PDBBind schema uses Complex as the row primitive instead of
    Example. Each Complex has at most one Ligand (complex_has_ligand),
    one Protein (complex_has_protein), one Pocket (complex_has_pocket),
    and a BindingMeasurement that carries the pK label.

    We turn every Complex into a v2 Example node with:
      - node_id = "Example::pdbbind::<pdb_id>"
      - label   = <pdb_id>
      - props   = {"label": <pact>, "target": <uniprot_or_pdb>, "source": "PDBBind"}
    and emit:
      - example_has_ligand (Example -> Ligand)
      - example_has_protein (Example -> Protein)
      - example_has_pocket  (Example -> Pocket)
      - example_from_source (Example -> DatasetSource::PDBBind)

    pK label extraction: look for `BindingMeasurement::*` nodes and their
    `binding_measurement_value` (if present); otherwise fall back to the
    pdbbind_index.parquet pK column via a future call. For now we leave
    label=0.0 if we can't read it from the BindingMeasurement node props.
    """
    # 1. Index BindingMeasurement nodes by their complex_id via the
    #    `complex_has_binding_measurement` edges (which we have in raw_edges
    #    even though we dropped them from the final v2 edge set).
    cmx_to_lig = (
        raw_edges.filter(pl.col("edge_type") == "complex_has_ligand")
        .select(pl.col("src").alias("complex_id"), pl.col("dst").alias("ligand_id"))
    )
    cmx_to_prot = (
        raw_edges.filter(pl.col("edge_type") == "complex_has_protein")
        .select(pl.col("src").alias("complex_id"), pl.col("dst").alias("protein_id"))
    )
    cmx_to_pocket = (
        raw_edges.filter(pl.col("edge_type") == "complex_has_pocket")
        .select(pl.col("src").alias("complex_id"), pl.col("dst").alias("pocket_id"))
    )
    cmx_to_bm = (
        raw_edges.filter(pl.col("edge_type") == "complex_has_binding_measurement")
        .select(pl.col("src").alias("complex_id"), pl.col("dst").alias("bm_id"))
    )
    cmx_to_src = (
        raw_edges.filter(pl.col("edge_type") == "complex_from_source")
        .select(pl.col("src").alias("complex_id"), pl.col("dst").alias("source_id"))
    )

    # 2. Identify Complex nodes from the *original* node dump (before _map_nodes
    #    dropped them) - they would have node_type == "Complex" there. The
    #    nodes argument here is post-mapping, so we need raw_nodes too.
    #    Instead we collect complex IDs from the binding edges.
    complex_ids = (
        pl.concat(
            [
                cmx_to_lig.select(pl.col("complex_id")),
                cmx_to_prot.select(pl.col("complex_id")),
                cmx_to_pocket.select(pl.col("complex_id")),
            ],
            how="vertical_relaxed",
        )
        ["complex_id"].unique().to_list()
    )

    if not complex_ids:
        return nodes, edges

    # 3. Look for BindingMeasurement nodes and extract pK from their props.
    #    Props JSON contains keys like {"value": 1.7, "unit": "uM",
    #    "p_value": 5.77, "p_name": "pKd", "comparator": "="}.
    bm_labels: dict[str, float] = {}
    bm_rows = raw_nodes.filter(pl.col("node_type") == "BindingMeasurement")
    if bm_rows.height:
        import json as _json
        for bm_id, props_str in zip(
            bm_rows["node_id"].to_list(),
            bm_rows["props"].to_list(),
        ):
            if not props_str:
                continue
            try:
                d = _json.loads(props_str)
            except (_json.JSONDecodeError, TypeError):
                continue
            pv = d.get("p_value")
            if pv is None:
                continue
            try:
                bm_labels[bm_id] = float(pv)
            except (TypeError, ValueError):
                continue

    # 4. Build Example nodes from each Complex. Use the complex_id as the
    #    pdb_id (v1 names them "Complex::<pdb_id>").
    example_node_ids: list[str] = []
    example_labels: list[str] = []
    example_props: list[str] = []

    cmx_to_lig_map = dict(zip(cmx_to_lig["complex_id"].to_list(), cmx_to_lig["ligand_id"].to_list()))
    cmx_to_prot_map = dict(zip(cmx_to_prot["complex_id"].to_list(), cmx_to_prot["protein_id"].to_list()))
    cmx_to_pocket_map = dict(zip(cmx_to_pocket["complex_id"].to_list(), cmx_to_pocket["pocket_id"].to_list()))
    cmx_to_bm_map = dict(zip(cmx_to_bm["complex_id"].to_list(), cmx_to_bm["bm_id"].to_list()))
    cmx_to_src_map = dict(zip(cmx_to_src["complex_id"].to_list(), cmx_to_src["source_id"].to_list()))

    new_edge_rows = []
    # Threshold to binarise pK for the pipeline's actives/inactives count.
    # PDBBind protocol convention: pK >= 6.0 (Ki/Kd/IC50 <= 1 uM) -> active.
    # The original pK stays in props["pk_value"] for follow-up regression.
    PK_BINARY_THRESHOLD = 6.0
    for cid in complex_ids:
        # Strip the "Complex::" prefix if present.
        pdb_id = cid.replace("Complex::", "") if cid.startswith("Complex::") else cid
        ex_id = f"Example::pdbbind::{pdb_id}"
        example_node_ids.append(ex_id)
        example_labels.append(pdb_id)
        # Look up the pK label via the BM node mapped to this complex.
        bm_id = cmx_to_bm_map.get(cid)
        pk = bm_labels.get(bm_id, 0.0) if bm_id else 0.0
        binary_label = 1.0 if pk >= PK_BINARY_THRESHOLD else 0.0
        example_props.append(
            f'{{"label": {binary_label}, "pk_value": {pk}, '
            f'"target": "{cmx_to_prot_map.get(cid, "")}", '
            f'"source": "PDBBind", "pdb_id": "{pdb_id}"}}'
        )
        if cid in cmx_to_lig_map:
            new_edge_rows.append({
                "src": ex_id, "dst": cmx_to_lig_map[cid],
                "edge_type": EdgeType.EXAMPLE_HAS_LIGAND.value,
                "props": "{}",
            })
        if cid in cmx_to_prot_map:
            new_edge_rows.append({
                "src": ex_id, "dst": cmx_to_prot_map[cid],
                "edge_type": EdgeType.EXAMPLE_HAS_PROTEIN.value,
                "props": "{}",
            })
        if cid in cmx_to_pocket_map:
            new_edge_rows.append({
                "src": ex_id, "dst": cmx_to_pocket_map[cid],
                "edge_type": EdgeType.EXAMPLE_HAS_POCKET.value,
                "props": "{}",
            })
        if cid in cmx_to_src_map:
            new_edge_rows.append({
                "src": ex_id, "dst": cmx_to_src_map[cid],
                "edge_type": EdgeType.EXAMPLE_FROM_SOURCE.value,
                "props": "{}",
            })

    n_examples = len(example_node_ids)
    example_nodes = pl.DataFrame({
        "node_id":   example_node_ids,
        "node_type": [NodeType.EXAMPLE.value] * n_examples,
        "label":     example_labels,
        "props":     example_props,
    })

    nodes = pl.concat([nodes, example_nodes], how="vertical_relaxed")
    if new_edge_rows:
        new_edges_df = pl.DataFrame(new_edge_rows)
        edges = pl.concat([edges, new_edges_df], how="vertical_relaxed")

    stats.deferred = (stats.deferred or []) + [
        f"pdbbind_examples_synthesized={n_examples}",
        "pdbbind_labels_default_zero_pending_BM_node_join",
    ]
    log.info("synthesized %d PDBBind Example nodes from Complex graph", n_examples)
    return nodes, edges


def _add_protein_cluster_edges(
    edges: pl.DataFrame,
    nodes: pl.DataFrame,
    processed: Path,
    stats: BuildStats,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add protein_in_cluster edges from pdbbind cluster parquets.

    v1 emits `pdbbind_protein_clusters_{30,50,90}.parquet` with columns
    (probably) (protein_id, cluster_id). We treat each cluster_id as a
    ProteinCluster node tagged with resolution = "30" | "50" | "90".

    The v2 schema names the 90% pocket-similarity edge weight
    `protein_cluster_90`. v1 happens to ship 30 / 50 / 90 not 30 / 40 /
    90; we keep all three but the 50% one will get the default weight
    until we re-cluster at 40%.
    """
    new_node_dfs: list[pl.DataFrame] = []
    new_edge_dfs: list[pl.DataFrame] = []
    counts: dict[str, int] = {}
    for res in ("30", "50", "90"):
        f = processed / f"pdbbind_protein_clusters_{res}.parquet"
        if not f.exists():
            counts[res] = 0
            stats.deferred = (stats.deferred or []) + [f"protein_clusters_{res}_missing"]
            continue
        df = pl.read_parquet(f)
        # try to find protein-id and cluster-id columns
        col_protein = next(
            (c for c in df.columns
             if c.lower() in ("protein_id", "pdb_id", "member",
                              "sequence_id", "seq_id")),
            None,
        )
        col_cluster = next(
            (c for c in df.columns
             if c.lower() in ("cluster_id", "cluster", "representative",
                              "rep_seq", "rep_seq_id")),
            None,
        )
        if not col_protein or not col_cluster:
            counts[res] = 0
            stats.deferred = (stats.deferred or []) + [
                f"protein_clusters_{res}_unknown_schema_cols={df.columns}"
            ]
            continue
        df2 = df.select(
            pl.col(col_protein).cast(pl.Utf8).alias("member_id"),
            pl.col(col_cluster).cast(pl.Utf8).alias("cluster_id"),
        )
        # Synthesise the ProteinCluster nodes (one per cluster_id).
        unique_clusters = df2["cluster_id"].unique().to_list()
        cluster_nodes = pl.DataFrame({
            "node_id":   [f"ProteinCluster::{res}::{c}" for c in unique_clusters],
            "node_type": [NodeType.PROTEIN_CLUSTER.value] * len(unique_clusters),
            "label":     [f"ProteinCluster::{res}::{c}" for c in unique_clusters],
            "props":     [f'{{"resolution":"{res}"}}'] * len(unique_clusters),
        })
        new_node_dfs.append(cluster_nodes)
        # Construct the protein_in_cluster edges (Protein -> ProteinCluster_<res>).
        edges_df = df2.with_columns(
            (pl.lit(f"ProteinCluster::{res}::") + pl.col("cluster_id").cast(pl.Utf8))
            .alias("dst"),
        ).select(
            pl.col("member_id").alias("src"),
            pl.col("dst"),
            pl.lit(EdgeType.PROTEIN_IN_CLUSTER.value).alias("edge_type"),
            pl.lit(f'{{"resolution":"{res}"}}').alias("props"),
        )
        new_edge_dfs.append(edges_df)
        counts[res] = edges_df.height
    stats.n_protein_cluster_edges = counts
    if new_node_dfs:
        nodes = pl.concat([nodes] + new_node_dfs, how="vertical_relaxed")
    if new_edge_dfs:
        edges = pl.concat([edges] + new_edge_dfs, how="vertical_relaxed")
    return nodes, edges


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_graph(
    output_dir: Path,
    *,
    corpus: str = "all",
    limit: int | None = None,
    hub_cfg: HubMitigationConfig | None = None,
) -> BuildStats:
    """Build the v2 graph parquets under `output_dir`.

    Parameters
    ----------
    output_dir
        Where to write v2_nodes.parquet, v2_edges.parquet, stats.csv.
    corpus
        "all" -> read v1's mvp2_nodes/edges (combined).
        Otherwise the per-corpus parquet name ("litpcba_ave", "dude",
        "dekois", "pdbbind").
    limit
        Optional row cap for smoke-testing.
    hub_cfg
        Hub-mitigation parameters; defaults to schema.HubMitigationConfig.
    """
    cfg = hub_cfg or HubMitigationConfig()
    processed = processed_dir()
    require_data_root()
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_name = "mvp2_nodes" if corpus == "all" else f"{corpus}_nodes"
    edges_name = "mvp2_edges" if corpus == "all" else f"{corpus}_edges"

    nodes_path = processed / f"{nodes_name}.parquet"
    edges_path = processed / f"{edges_name}.parquet"
    for p in (nodes_path, edges_path):
        if not p.exists():
            raise FileNotFoundError(p)

    stats = BuildStats()
    t0 = time.perf_counter()

    # Read EAGERLY so we only hit NFS once per file - critical on slow/loaded
    # NFS storage where mmap'd scan_parquet causes many small page faults.
    nodes = pl.read_parquet(nodes_path)
    edges = pl.read_parquet(edges_path)
    raw_edges = edges  # Keep the pre-mapping copies for PDBBind Complex synthesis.
    raw_nodes = nodes
    if limit:
        nodes = nodes.head(limit)
        edges = edges.head(limit)
        raw_edges = raw_edges.head(limit)
        raw_nodes = raw_nodes.head(limit)

    stats.n_nodes_in = nodes.height
    stats.n_edges_in = edges.height
    log.info("read %d rows from %s", stats.n_nodes_in, nodes_path.name)
    log.info("read %d rows from %s", stats.n_edges_in, edges_path.name)

    nodes, dropped_n = _map_nodes(nodes)
    edges, dropped_e = _map_edges(edges)
    stats.n_nodes_dropped = dropped_n
    stats.n_edges_dropped = dropped_e
    log.info("mapped: n_nodes=%d (-%d), n_edges=%d (-%d)",
             nodes.height, dropped_n, edges.height, dropped_e)

    # PDBBind has no native Example nodes - synthesize them from Complex.
    if corpus == "pdbbind":
        nodes, edges = _synthesize_pdbbind_examples(nodes, edges, raw_edges, raw_nodes, stats)
        log.info("after PDBBind Example synthesis: n_nodes=%d, n_edges=%d",
                 nodes.height, edges.height)

    nodes, edges = _add_protein_cluster_edges(edges, nodes, processed, stats)
    log.info("after cluster edges: n_nodes=%d, n_edges=%d", nodes.height, edges.height)

    nodes, edges, n_trivial = _drop_trivial_scaffolds(nodes, edges, cfg)
    stats.n_trivial_scaffolds_dropped = n_trivial
    log.info("dropped %d trivial scaffolds", n_trivial)

    nodes, edges, n_hubs = _shard_hub_nodes(nodes, edges, cfg)
    stats.n_hub_nodes_sharded = n_hubs
    log.info("flagged %d hub nodes", n_hubs)

    # Already eager DataFrames at this point.
    nodes_df = nodes
    edges_df = edges
    stats.n_nodes_out = nodes_df.height
    stats.n_edges_out = edges_df.height
    stats.nodes_by_type = dict(
        nodes_df.group_by("node_type").len().sort("len", descending=True).iter_rows()
    )
    stats.edges_by_type = dict(
        edges_df.group_by("edge_type").len().sort("len", descending=True).iter_rows()
    )
    # Things we cannot compute on this box without an encoder.
    stats.deferred = (stats.deferred or []) + [
        "pocket_similar_edges_need_ESM_IF1_or_equivalent",
        "time_overlap_edges_need_ChEMBL_dates",
        "example_from_assay_needs_chembl_assay_join",
        "example_from_publication_needs_chembl_document_join",
    ]

    nodes_out = output_dir / "v2_nodes.parquet"
    edges_out = output_dir / "v2_edges.parquet"
    stats_out = output_dir / "stats.csv"

    nodes_df.write_parquet(nodes_out)
    edges_df.write_parquet(edges_out)
    pl.DataFrame(stats.to_csv_rows()).write_csv(stats_out)

    log.info(
        "v2 graph: %s nodes, %s edges, wrote to %s (%.1fs)",
        stats.n_nodes_out,
        stats.n_edges_out,
        output_dir,
        time.perf_counter() - t0,
    )
    return stats


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--corpus", default="all",
                   choices=["all", "litpcba_ave", "dude", "dekois", "pdbbind"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    stats = build_graph(
        output_dir=args.output_dir,
        corpus=args.corpus,
        limit=args.limit,
    )
    print(f"nodes_out={stats.n_nodes_out} edges_out={stats.n_edges_out}")


if __name__ == "__main__":
    _cli()
