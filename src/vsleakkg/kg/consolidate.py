"""Consolidate the raw KG into the canonical audit-ready schema.

Reads the raw KG produced by `vsleakkg.build_kg`:

  data/processed/kg_nodes.parquet
  data/processed/kg_edges.parquet
  data/processed/protein_clusters_{30,50,90}.parquet  (sequence-axis anchor,
                                                       optional)

Applies:
  1. Schema mapping     — collapse corpus-level type names (e.g. ChEMBLAssay
                          → Assay, ProteinTarget → Protein) onto the canonical
                          set defined in `vsleakkg.kg.schema`.
  2. Lossy node drop    — discard pure scaffolding (ChEMBLActivity, Split,
                          LabelType, DatabaseRelease) that the audit doesn't
                          need.
  3. Protein clustering — emit ProteinCluster nodes + protein_in_cluster edges
                          from the optional sequence-clustered parquets.
  4. Hub mitigation     — flag any node with degree > HubMitigationConfig.
                          degree_cap (default 1000) as `is_hub`.
  5. Trivial scaffold   — drop scaffolds with ≤ trivial_scaffold_max_atoms
                          (default 6) heavy atoms.

Outputs:

  outputs/kg/canonical_nodes.parquet
  outputs/kg/canonical_edges.parquet
  outputs/kg/stats.csv

CLI:

    python -m vsleakkg.kg.consolidate \\
        --output-dir outputs/kg \\
        [--corpus litpcba_ave|dude|dekois|bigbind|bayesbind|all]   # default all
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
# Corpus-level → canonical schema mapping tables
# ---------------------------------------------------------------------------
# `build_kg` emits the raw KG using the corpus-level node/edge type names
# inherited from the per-corpus loaders (`Ligand`, `ProteinTarget`,
# `ChEMBLAssay`, ...). The canonical schema (`vsleakkg.kg.schema`) collapses
# semantically-equivalent types onto a smaller set. These maps drive the
# rewrite, with the side effect that certain pure-scaffolding node/edge types
# are dropped entirely.

CORPUS_TO_CANONICAL_NODE_TYPE: dict[str, str] = {
    "Example": NodeType.EXAMPLE.value,
    "Ligand": NodeType.LIGAND.value,
    "Scaffold": NodeType.SCAFFOLD.value,
    "Protein": NodeType.PROTEIN.value,
    "ProteinTarget": NodeType.PROTEIN.value,
    "ChEMBLAssay": NodeType.ASSAY.value,
    "Assay": NodeType.ASSAY.value,                # identity: BindingDB-emitted Assay-like records
    "ChEMBLDocument": NodeType.PUBLICATION.value,
    "Publication": NodeType.PUBLICATION.value,    # identity: BindingDB-emitted PMID/DOI publications
    "DatasetSource": NodeType.DATASET_SOURCE.value,
    "DecoyProtocol": NodeType.DECOY_PROTOCOL.value,
}

# Corpus-level node types that are absorbed elsewhere or are pure scaffolding.
DROPPED_NODES: frozenset[str] = frozenset({
    "ChEMBLActivity",        # absorbed into Example via label/label_type
    "AffinityType",          # static lookup table
    "LabelType",             # static lookup table
    "DatabaseRelease",       # version metadata
    "Split",                 # corpus-level split labels; canonical schema emits
                             # partition assignments separately
})

CORPUS_TO_CANONICAL_EDGE_TYPE: dict[str, str] = {
    "example_has_ligand": EdgeType.EXAMPLE_HAS_LIGAND.value,
    "example_targets_protein": EdgeType.EXAMPLE_HAS_PROTEIN.value,
    "example_from_source": EdgeType.EXAMPLE_FROM_SOURCE.value,
    "ligand_has_scaffold": EdgeType.LIGAND_SCAFFOLD.value,
    "ligand_similar_to_ligand": EdgeType.LIGAND_SIMILAR.value,
    "ligand_similar": EdgeType.LIGAND_SIMILAR.value,    # ligand_similarity.py emits this name directly
    "ligand_fingerprint_exact": EdgeType.LIGAND_FINGERPRINT_EXACT.value,
    "same_inchikey_as": EdgeType.LIGAND_EXACT.value,
    "same_parent_inchikey_as": EdgeType.LIGAND_PARENT_EXACT.value,
    "example_uses_decoy_protocol": EdgeType.SOURCE_DECOY_PROTOCOL.value,
}
# BindingDB enrichment edges (bdb_lig -> Publication / Protein / Assay /
# bdb_rec -> Ligand / Protein) are NOT mapped here: they form 2-hop paths
# through `bdb_lig` from benchmark Example -> Ligand -> bdb_lig, which the
# canonical single-edge axis schema can't represent. They remain in the raw
# kg_edges parquet for downstream graph-traversal queries that want the
# full BindingDB provenance.

DROPPED_EDGES: frozenset[str] = frozenset({
    "example_in_split",
    "example_has_label_type",
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
    """Map corpus node_type to canonical node_type. Drop scaffolding-only types."""
    keep = list(CORPUS_TO_CANONICAL_NODE_TYPE.keys())
    n_in = nodes.height
    kept = nodes.filter(pl.col("node_type").is_in(keep)).with_columns(
        pl.col("node_type").replace(CORPUS_TO_CANONICAL_NODE_TYPE).alias("node_type")
    )
    return kept, n_in - kept.height


def _map_edges(edges: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Map corpus edge_type to canonical edge_type. Drop scaffolding-only edges."""
    keep = list(CORPUS_TO_CANONICAL_EDGE_TYPE.keys())
    n_in = edges.height
    kept = edges.filter(pl.col("edge_type").is_in(keep)).with_columns(
        pl.col("edge_type").replace(CORPUS_TO_CANONICAL_EDGE_TYPE).alias("edge_type")
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


def _add_protein_cluster_edges(
    edges: pl.DataFrame,
    nodes: pl.DataFrame,
    processed: Path,
    stats: BuildStats,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add protein_in_cluster edges from sequence-clustered parquets.

    Reads `protein_clusters_{30,50,90}.parquet` with columns (protein_id,
    cluster_id). Each cluster_id becomes a ProteinCluster node tagged with
    resolution = "30" | "50" | "90".

    The canonical schema weights are tuned for 30 / 50 / 90% sequence identity.
    Missing parquets are recorded in `stats.deferred` for the audit report.
    """
    new_node_dfs: list[pl.DataFrame] = []
    new_edge_dfs: list[pl.DataFrame] = []
    counts: dict[str, int] = {}
    for res in ("30", "50", "90"):
        f = processed / f"protein_clusters_{res}.parquet"
        if not f.exists():
            counts[res] = 0
            stats.deferred = (stats.deferred or []) + [f"protein_clusters_{res}_missing"]
            continue
        df = pl.read_parquet(f)
        # try to find protein-id and cluster-id columns
        col_protein = next(
            (c for c in df.columns
             if c.lower() in ("accession", "protein_id", "pdb_id", "member",
                              "sequence_id", "seq_id", "uniprot")),
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
            # Prefix accession with `protein:` so the edge src matches the
            # Protein node id format used by the BindingDB enrichment step
            # in task_build_kg (`protein:<UniProt>`).
            (pl.lit("protein:") + pl.col(col_protein).cast(pl.Utf8)).alias("member_id"),
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


def _wire_reference_provenance(
    raw_edges: pl.DataFrame,
    canonical_edges: pl.DataFrame,
    log: logging.Logger,
) -> pl.DataFrame:
    """Synthesize Example -> Assay / Publication / Protein direct edges by
    collapsing multi-hop paths through the raw KG's ChEMBL Activity and
    BindingDB record subgraphs.

    The raw kg_edges parquet carries the full reference-DB provenance
    (chembl_activity_has_assay / _has_document / _has_target,
    bindingdb_ligand_in_publication / _targets_protein,
    bindingdb_record_has_ligand / _has_protein), but the canonical schema
    only encodes single-edge axis relationships. We compose direct edges
    here so the audit can traverse the assay / publication axes without
    expanding multi-hop paths every query.

    Returns the canonical edges with new synthetic edges concatenated.
    Caller is responsible for downstream dedup.
    """
    import json as _json
    # All synthesis chains start with Example -> benchmark Ligand.
    ex_lig = (raw_edges.filter(pl.col("edge_type") == "example_has_ligand")
              .select([pl.col("src").alias("example_id"),
                       pl.col("dst").alias("bench_lid")])
              .unique())
    n_synth_pub = n_synth_assay = n_synth_prot = 0

    # ---- ChEMBL chain ----
    # Build the chain in *aggregated* steps so we never materialise the full
    # Cartesian explosion. Each pair set is deduped before the next join.
    bench_to_chembl = (raw_edges.filter(
            pl.col("edge_type") == "benchmark_ligand_same_inchikey_as_chembl_ligand")
        .select([pl.col("src").alias("bench_lid"),
                 pl.col("dst").alias("chembl_lid")])
        .unique())
    if bench_to_chembl.height:
        chembl_lig_to_act = (raw_edges.filter(
                pl.col("edge_type") == "chembl_activity_has_ligand")
            .select([pl.col("dst").alias("chembl_lid"),
                     pl.col("src").alias("chembl_act_id")])
            .unique())
        log.info("  chembl chain: %d bench->chembl_lig, %d chembl_lig->act",
                 bench_to_chembl.height, chembl_lig_to_act.height)

        # Activity -> Document, deduped first to drop redundant pairs.
        act_doc = (raw_edges.filter(
                pl.col("edge_type") == "chembl_activity_has_document")
            .select([pl.col("src").alias("chembl_act_id"),
                     pl.col("dst").alias("chembl_doc_id")])
            .unique())
        if act_doc.height:
            chembl_lig_to_doc = (chembl_lig_to_act
                                 .join(act_doc, on="chembl_act_id", how="inner")
                                 .select(["chembl_lid", "chembl_doc_id"]).unique())
            bench_to_doc = (bench_to_chembl
                            .join(chembl_lig_to_doc, on="chembl_lid", how="inner")
                            .select(["bench_lid", "chembl_doc_id"]).unique())
            log.info("  chembl: %d unique chembl_lig->doc, %d bench->doc",
                     chembl_lig_to_doc.height, bench_to_doc.height)
            ex_to_doc = (ex_lig
                         .join(bench_to_doc, on="bench_lid", how="inner")
                         .select([pl.col("example_id").alias("src"),
                                  pl.col("chembl_doc_id").alias("dst")])
                         .unique())
            ex_to_doc = ex_to_doc.with_columns([
                pl.lit(EdgeType.EXAMPLE_FROM_PUBLICATION.value).alias("edge_type"),
                pl.lit(_json.dumps({"source": "ChEMBL"})).alias("props"),
            ])
            canonical_edges = pl.concat([canonical_edges, ex_to_doc],
                                        how="vertical_relaxed")
            n_synth_pub += ex_to_doc.height

        # Activity -> Assay, same aggregated pattern.
        act_asy = (raw_edges.filter(
                pl.col("edge_type") == "chembl_activity_has_assay")
            .select([pl.col("src").alias("chembl_act_id"),
                     pl.col("dst").alias("chembl_asy_id")])
            .unique())
        if act_asy.height:
            chembl_lig_to_asy = (chembl_lig_to_act
                                 .join(act_asy, on="chembl_act_id", how="inner")
                                 .select(["chembl_lid", "chembl_asy_id"]).unique())
            bench_to_asy = (bench_to_chembl
                            .join(chembl_lig_to_asy, on="chembl_lid", how="inner")
                            .select(["bench_lid", "chembl_asy_id"]).unique())
            log.info("  chembl: %d unique chembl_lig->asy, %d bench->asy",
                     chembl_lig_to_asy.height, bench_to_asy.height)
            ex_to_asy = (ex_lig
                         .join(bench_to_asy, on="bench_lid", how="inner")
                         .select([pl.col("example_id").alias("src"),
                                  pl.col("chembl_asy_id").alias("dst")])
                         .unique())
            ex_to_asy = ex_to_asy.with_columns([
                pl.lit(EdgeType.EXAMPLE_FROM_ASSAY.value).alias("edge_type"),
                pl.lit(_json.dumps({"source": "ChEMBL"})).alias("props"),
            ])
            canonical_edges = pl.concat([canonical_edges, ex_to_asy],
                                        how="vertical_relaxed")
            n_synth_assay += ex_to_asy.height

    # ---- BindingDB chain ----
    bench_to_bdb = (raw_edges.filter(
            pl.col("edge_type") == "benchmark_ligand_same_inchikey_as_bindingdb_ligand")
        .select([pl.col("src").alias("bench_lid"),
                 pl.col("dst").alias("bdb_lid")])
        .unique())
    if bench_to_bdb.height:
        # bdb_lig -> Publication
        bdb_pub = (raw_edges.filter(
                pl.col("edge_type") == "bindingdb_ligand_in_publication")
            .select([pl.col("src").alias("bdb_lid"),
                     pl.col("dst").alias("pub_id")])
            .unique())
        if bdb_pub.height:
            ex_to_bdb_pub = (ex_lig
                             .join(bench_to_bdb, on="bench_lid", how="inner")
                             .join(bdb_pub, on="bdb_lid", how="inner")
                             .select([pl.col("example_id").alias("src"),
                                      pl.col("pub_id").alias("dst")])
                             .unique())
            ex_to_bdb_pub = ex_to_bdb_pub.with_columns([
                pl.lit(EdgeType.EXAMPLE_FROM_PUBLICATION.value).alias("edge_type"),
                pl.lit(_json.dumps({"source": "BindingDB"})).alias("props"),
            ])
            canonical_edges = pl.concat([canonical_edges, ex_to_bdb_pub],
                                        how="vertical_relaxed")
            n_synth_pub += ex_to_bdb_pub.height

        # bdb_lig -> Protein (UniProt)
        bdb_prot = (raw_edges.filter(
                pl.col("edge_type") == "bindingdb_ligand_targets_protein")
            .select([pl.col("src").alias("bdb_lid"),
                     pl.col("dst").alias("prot_id")])
            .unique())
        if bdb_prot.height:
            ex_to_bdb_prot = (ex_lig
                              .join(bench_to_bdb, on="bench_lid", how="inner")
                              .join(bdb_prot, on="bdb_lid", how="inner")
                              .select([pl.col("example_id").alias("src"),
                                       pl.col("prot_id").alias("dst")])
                              .unique())
            ex_to_bdb_prot = ex_to_bdb_prot.with_columns([
                pl.lit(EdgeType.EXAMPLE_HAS_PROTEIN.value).alias("edge_type"),
                pl.lit(_json.dumps({"source": "BindingDB"})).alias("props"),
            ])
            canonical_edges = pl.concat([canonical_edges, ex_to_bdb_prot],
                                        how="vertical_relaxed")
            n_synth_prot += ex_to_bdb_prot.height

        # bdb_lig -> BindingDB record (Assay-like)
        bdb_rec = (raw_edges.filter(
                pl.col("edge_type") == "bindingdb_record_has_ligand")
            .select([pl.col("dst").alias("bdb_lid"),
                     pl.col("src").alias("bdb_rec_id")])
            .unique())
        if bdb_rec.height:
            ex_to_bdb_rec = (ex_lig
                             .join(bench_to_bdb, on="bench_lid", how="inner")
                             .join(bdb_rec, on="bdb_lid", how="inner")
                             .select([pl.col("example_id").alias("src"),
                                      pl.col("bdb_rec_id").alias("dst")])
                             .unique())
            ex_to_bdb_rec = ex_to_bdb_rec.with_columns([
                pl.lit(EdgeType.EXAMPLE_FROM_ASSAY.value).alias("edge_type"),
                pl.lit(_json.dumps({"source": "BindingDB"})).alias("props"),
            ])
            canonical_edges = pl.concat([canonical_edges, ex_to_bdb_rec],
                                        how="vertical_relaxed")
            n_synth_assay += ex_to_bdb_rec.height

    log.info("wire_ref_provenance: synth %d publication + %d assay + %d protein edges",
             n_synth_pub, n_synth_assay, n_synth_prot)
    return canonical_edges


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def consolidate(
    output_dir: Path,
    *,
    corpus: str = "all",
    limit: int | None = None,
    hub_cfg: HubMitigationConfig | None = None,
) -> BuildStats:
    """Consolidate the raw KG into the canonical schema parquets.

    Parameters
    ----------
    output_dir
        Where to write canonical_nodes.parquet, canonical_edges.parquet,
        stats.csv.
    corpus
        "all" -> read the merged kg_nodes/kg_edges from `build_kg`.
        Otherwise the per-corpus parquet name ("litpcba_ave", "dude",
        "dekois", "bigbind", "bayesbind").
    limit
        Optional row cap for smoke-testing.
    hub_cfg
        Hub-mitigation parameters; defaults to schema.HubMitigationConfig.
    """
    cfg = hub_cfg or HubMitigationConfig()
    processed = processed_dir()
    require_data_root()
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_name = "kg_nodes" if corpus == "all" else f"{corpus}_nodes"
    edges_name = "kg_edges" if corpus == "all" else f"{corpus}_edges"

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
    raw_edges_full = edges       # preserved for reference-provenance wiring below
    if limit:
        nodes = nodes.head(limit)
        edges = edges.head(limit)
        raw_edges_full = raw_edges_full.head(limit)

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

    # B1+B2+B3: synthesize Example -> Publication / Assay / Protein direct edges
    # by collapsing multi-hop ChEMBL + BindingDB provenance chains. Done before
    # the cluster + dangling steps so the new edges participate in dangling
    # detection and orphan accounting.
    edges = _wire_reference_provenance(raw_edges_full, edges, log)

    nodes, edges = _add_protein_cluster_edges(edges, nodes, processed, stats)
    log.info("after cluster edges: n_nodes=%d, n_edges=%d", nodes.height, edges.height)

    nodes, edges, n_trivial = _drop_trivial_scaffolds(nodes, edges, cfg)
    stats.n_trivial_scaffolds_dropped = n_trivial
    log.info("dropped %d trivial scaffolds", n_trivial)

    nodes, edges, n_hubs = _shard_hub_nodes(nodes, edges, cfg)
    stats.n_hub_nodes_sharded = n_hubs
    log.info("flagged %d hub nodes", n_hubs)

    # Prune dangling edges (cluster edges typically dangle when corpus-level
    # Protein ids don't match the UniProt-based cluster member ids).
    _ids = nodes.select("node_id")
    n_before = edges.height
    edges = (edges.join(_ids.rename({"node_id": "src"}), on="src", how="semi")
                  .join(_ids.rename({"node_id": "dst"}), on="dst", how="semi"))
    pruned = n_before - edges.height
    if pruned:
        log.info("pruned %d dangling edges", pruned)
        stats.deferred = (stats.deferred or []) + [f"pruned_dangling_edges={pruned}"]

    # Universal orphan drop: any node with degree 0 after the dangling-edge
    # prune is removed. This covers Protein/ProteinCluster (cluster member ids
    # that don't match KG protein ids), Assay/Publication (when reference-DB
    # wiring fails to reach them), and any other isolated node accumulated by
    # the build. The five small-cardinality "structure" types (DatasetSource,
    # DecoyProtocol, plus everything Example-side) are excluded so a corpus
    # with zero edges still keeps its DatasetSource pin.
    touched_ids = pl.concat(
        [edges.select(pl.col("src").alias("node_id")),
         edges.select(pl.col("dst").alias("node_id"))],
        how="vertical_relaxed",
    ).unique()
    keep_anyway = {"DatasetSource", "DecoyProtocol"}
    keep_pinned = nodes.filter(pl.col("node_type").is_in(list(keep_anyway)))
    drop_candidates = nodes.filter(~pl.col("node_type").is_in(list(keep_anyway)))
    kept_via_edges = drop_candidates.join(touched_ids, on="node_id", how="semi")
    n_before_n = nodes.height
    nodes = pl.concat([keep_pinned, kept_via_edges], how="vertical_relaxed")
    dropped_orphans = n_before_n - nodes.height
    if dropped_orphans:
        by_type = (drop_candidates
                   .join(touched_ids, on="node_id", how="anti")
                   .group_by("node_type").len()
                   .sort("len", descending=True))
        log.info("dropped %d orphan nodes: %s", dropped_orphans,
                 dict(by_type.iter_rows()))
        stats.deferred = (stats.deferred or []) + [f"dropped_orphans={dropped_orphans}"]

    # Dedup (src, dst, edge_type) — multiple raw sources can emit the same
    # logical edge (e.g. per-corpus ligand_similar_to_ligand + D5's
    # ligand_similar both collapse to LIGAND_SIMILAR). Keep one row per triple.
    n_before_e = edges.height
    edges = edges.unique(subset=["src", "dst", "edge_type"], keep="first")
    deduped = n_before_e - edges.height
    if deduped:
        log.info("deduped %d redundant edges on (src, dst, edge_type)", deduped)
        stats.deferred = (stats.deferred or []) + [f"deduped_redundant_edges={deduped}"]

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
        "time_overlap_edges_need_ChEMBL_dates",
        "example_from_assay_needs_chembl_assay_join",
        "example_from_publication_needs_chembl_document_join",
    ]

    nodes_out = output_dir / "canonical_nodes.parquet"
    edges_out = output_dir / "canonical_edges.parquet"
    stats_out = output_dir / "stats.csv"

    nodes_df.write_parquet(nodes_out)
    edges_df.write_parquet(edges_out)
    pl.DataFrame(stats.to_csv_rows()).write_csv(stats_out)

    log.info(
        "canonical KG: %s nodes, %s edges, wrote to %s (%.1fs)",
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
                   choices=["all", "litpcba_ave", "dude", "dekois", "bigbind", "bayesbind"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    stats = consolidate(
        output_dir=args.output_dir,
        corpus=args.corpus,
        limit=args.limit,
    )
    print(f"nodes_out={stats.n_nodes_out} edges_out={stats.n_edges_out}")


if __name__ == "__main__":
    _cli()
