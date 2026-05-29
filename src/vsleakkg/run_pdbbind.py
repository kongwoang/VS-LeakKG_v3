"""PDBBind / PBDBind processing orchestrator.

Implements the post-MVP-1 PDBBind pipeline: parse index + structures, audit
chemical leakage, build a graph, and merge with the existing MVP-1 graph by
canonical SMILES and InChIKey.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from vsleakkg import chem as vc
from vsleakkg import build_graph as vb
from vsleakkg import load_pdbbind

PROJECT_ROOT = Path("D:/hoangpc/VS-LeakKG")
RAW          = PROJECT_ROOT / "data" / "raw" / "PBDBind"
EXTRACTED    = RAW / "extracted"
PL_ROOT      = EXTRACTED / "P-L"
INDEX_ROOT   = EXTRACTED / "index"
PROCESSED    = PROJECT_ROOT / "data" / "processed"
TABLES       = PROJECT_ROOT / "outputs" / "tables"
REPORTS      = PROJECT_ROOT / "outputs" / "reports"
LOGS         = PROJECT_ROOT / "outputs" / "logs"
DISK_LOG     = LOGS / "pdbbind_disk_usage.log"
RUN_LOG      = LOGS / "pdbbind_processing.log"

for d in (PROCESSED, TABLES, REPORTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(RUN_LOG, mode="a", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("vsleakkg.pdbbind")


def log_step(event: str, target: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [f"==== {ts} ====", f"event: {event}", f"target: {target}",
             f"cwd: {os.getcwd()}"]
    try:
        u = shutil.disk_usage(PROJECT_ROOT)
        lines.append(f"  drive D: used={u.used/1024**3:.2f}GB "
                     f"free={u.free/1024**3:.2f}GB")
    except OSError:
        pass
    total = 0
    for p in PROJECT_ROOT.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    lines.append(f"-- project size: {total/1024**3:.2f} GB")
    lines.append("")
    DISK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DISK_LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------- worker functions (top-level for spawn) ---------

def _process_complex(args: tuple) -> dict:
    """Worker: parse one complex's ligand + protein. Returns a dict of fields
    that the main process will accumulate into a Polars frame."""
    pdb_id, year_bucket, lig_mol2, lig_sdf, pocket_pdb, protein_pdb = args
    lm2 = Path(lig_mol2) if lig_mol2 else None
    lsdf = Path(lig_sdf) if lig_sdf else None
    pkt = Path(pocket_pdb) if pocket_pdb else None
    prt = Path(protein_pdb) if protein_pdb else None

    canon, ik, scaf, lig_atoms, lig_fmt, lig_ok = load_pdbbind.parse_ligand(lm2, lsdf)
    fp_bytes = vc.ecfp_bytes(canon) if canon else None
    prot = load_pdbbind.parse_protein_pdb(prt)
    pocket_atoms = load_pdbbind.parse_pdb_atom_count(pkt) if pkt else 0

    return {
        "pdb_id": pdb_id,
        "year_bucket": year_bucket,
        "ligand_format_used": lig_fmt,
        "ligand_parse_ok": lig_ok,
        "ligand_smiles_canonical": canon,
        "ligand_inchikey": ik,
        "ligand_scaffold_smiles": scaf,
        "ligand_n_atoms": lig_atoms,
        "ligand_fp_bytes": fp_bytes,
        "protein_parse_ok": prot["parse_ok"],
        "protein_n_chains": prot["n_chains"],
        "protein_n_residues": prot["n_residues"],
        "protein_n_atoms": prot["n_atoms"],
        "protein_sequence_concat": prot["sequence_concat"],
        "protein_chains": ",".join(prot["chains"]) if prot["chains"] else None,
        "has_pocket_file": pkt is not None,
        "pocket_n_atoms": pocket_atoms,
        "has_ligand_mol2": lm2 is not None,
        "has_ligand_sdf": lsdf is not None,
        "has_protein_pdb": prt is not None,
    }


def parallel_process(complexes, workers: int, chunksize: int = 64) -> pl.DataFrame:
    args = [
        (c.pdb_id, c.year_bucket,
         str(c.ligand_mol2) if c.ligand_mol2 else None,
         str(c.ligand_sdf)  if c.ligand_sdf  else None,
         str(c.pocket_pdb)  if c.pocket_pdb  else None,
         str(c.protein_pdb) if c.protein_pdb else None)
        for c in complexes
    ]
    rows: List[dict] = []
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        for i, r in enumerate(pool.imap(_process_complex, args, chunksize=chunksize), 1):
            rows.append(r)
            if i % 2000 == 0 or i == len(args):
                log.info("processed %d / %d complexes", i, len(args))
    # Force string-typed columns for Polars where None is the only value sometimes.
    return pl.DataFrame(rows)


# --------- main pipeline ---------

def task_parse_index() -> pl.DataFrame:
    log_step("pre_step", "parse_index")
    idx_path = INDEX_ROOT / "INDEX_general_PL.2020R1.lst"
    if not idx_path.exists():
        raise FileNotFoundError(f"PDBBind PL index missing: {idx_path}")
    df = load_pdbbind.parse_pl_index(idx_path)
    df.write_parquet(PROCESSED / "pdbbind_index.parquet")
    aff_stats = (
        df.filter(pl.col("affinity_type").is_not_null())
          .group_by(["affinity_type", "unit", "comparator"])
          .agg(pl.len().alias("n")).sort("n", descending=True)
    )
    aff_summary = (df.group_by("affinity_parse_ok").agg(pl.len().alias("n"))
                   .sort("affinity_parse_ok"))
    n_total = df.height
    n_parsed = int(df["affinity_parse_ok"].sum())
    (REPORTS / "pdbbind_index_summary.md").write_text(
        "# PDBBind PL index — summary\n\n"
        f"Generated by `vsleakkg.run_pdbbind` at {datetime.now(timezone.utc).isoformat()}.\n\n"
        f"- complexes in index: **{n_total}**\n"
        f"- affinity parsed:    **{n_parsed}** ({n_parsed/n_total:.1%})\n"
        f"- NMR-only complexes: **{int(df['is_nmr'].sum())}**\n\n"
        "## Affinity type × unit × comparator (top combinations)\n\n"
        + aff_stats.head(30).to_pandas().to_string(index=False) + "\n\n"
        "## Affinity types\n\n"
        + df.group_by("affinity_type").agg(pl.len().alias("n"))
            .sort("n", descending=True).to_pandas().to_string(index=False) + "\n",
        encoding="utf-8")
    log_step("post_step", f"parse_index n={n_total}")
    return df


def task_parse_structures(workers: int) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    log_step("pre_step", "parse_structures")
    cached = PROCESSED / "pdbbind_complexes.parquet"
    if cached.exists():
        complexes = pl.read_parquet(cached)
        log.info("PDBBind: cached parsed complexes %s (n=%d)", cached, complexes.height)
        log_step("post_step", "parse_structures cached")
    else:
        files = load_pdbbind.discover_complexes(PL_ROOT)
        log.info("PDBBind: discovered %d complex dirs under %s", len(files), PL_ROOT)
        if not files:
            raise RuntimeError(f"No complex dirs under {PL_ROOT}")
        t0 = time.time()
        complexes = parallel_process(files, workers=workers)
        log.info("PDBBind: parsed %d complexes in %.1fs", complexes.height, time.time()-t0)
        complexes.write_parquet(cached)
        log_step("post_step", f"parse_structures n={complexes.height}")

    # Ligand-level frame (one row per unique canonical SMILES, but keep ligand
    # appearances joined back via complex_pdb_id list).
    ligands = (
        complexes.filter(pl.col("ligand_parse_ok"))
                 .select(["ligand_smiles_canonical", "ligand_inchikey",
                          "ligand_scaffold_smiles", "ligand_fp_bytes",
                          "ligand_n_atoms", "pdb_id"])
                 .group_by("ligand_smiles_canonical")
                 .agg([
                     pl.col("ligand_inchikey").first().alias("inchikey"),
                     pl.col("ligand_scaffold_smiles").first().alias("scaffold_smiles"),
                     pl.col("ligand_fp_bytes").first().alias("fp_bytes"),
                     pl.col("ligand_n_atoms").first().alias("n_atoms"),
                     pl.col("pdb_id").alias("pdb_ids"),
                     pl.len().alias("n_complexes"),
                 ])
                 .rename({"ligand_smiles_canonical": "canonical_smiles"})
                 .sort("n_complexes", descending=True)
    )
    ligands.write_parquet(PROCESSED / "pdbbind_ligands.parquet")
    log.info("PDBBind: %d unique canonical ligand SMILES", ligands.height)

    # Protein-level: dedupe by sequence_concat sha256 ("exact-sequence cluster").
    proteins = complexes.filter(pl.col("protein_parse_ok"))
    proteins = proteins.with_columns(
        pl.col("protein_sequence_concat").map_elements(
            lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest() if s else None,
            return_dtype=pl.Utf8, skip_nulls=False,
        ).alias("seq_sha256")
    )
    proteins = (proteins.select(["seq_sha256", "protein_sequence_concat",
                                  "protein_n_chains", "protein_n_residues",
                                  "protein_n_atoms", "pdb_id"])
                 .group_by("seq_sha256")
                 .agg([
                     pl.col("protein_sequence_concat").first().alias("sequence_concat"),
                     pl.col("protein_n_chains").first().alias("n_chains"),
                     pl.col("protein_n_residues").first().alias("n_residues"),
                     pl.col("protein_n_atoms").first().alias("n_atoms"),
                     pl.col("pdb_id").alias("pdb_ids"),
                     pl.len().alias("n_complexes"),
                 ])
                 .sort("n_complexes", descending=True))
    proteins.write_parquet(PROCESSED / "pdbbind_proteins.parquet")
    log.info("PDBBind: %d unique exact-sequence protein clusters", proteins.height)

    # Structure parse report.
    n = complexes.height
    n_lig_ok = int(complexes["ligand_parse_ok"].sum())
    n_prt_ok = int(complexes["protein_parse_ok"].sum())
    n_with_pocket = int(complexes["has_pocket_file"].sum())
    n_lig_mol2 = int(complexes["has_ligand_mol2"].sum())
    n_lig_sdf  = int(complexes["has_ligand_sdf"].sum())
    fmt_counts = (complexes.filter(pl.col("ligand_parse_ok"))
                  .group_by("ligand_format_used")
                  .agg(pl.len().alias("n")))
    (REPORTS / "pdbbind_structure_parse_report.md").write_text(
        "# PDBBind structure parse report\n\n"
        f"Generated by `vsleakkg.run_pdbbind` at {datetime.now(timezone.utc).isoformat()}.\n\n"
        f"- complexes:                       **{n}**\n"
        f"- ligand_parse_ok:                 **{n_lig_ok}** ({n_lig_ok/n:.1%})\n"
        f"- protein_parse_ok:                **{n_prt_ok}** ({n_prt_ok/n:.1%})\n"
        f"- complexes with pocket.pdb:       **{n_with_pocket}**\n"
        f"- complexes with ligand mol2:      **{n_lig_mol2}**\n"
        f"- complexes with ligand sdf:       **{n_lig_sdf}**\n"
        f"- unique canonical ligand SMILES:  **{ligands.height}**\n"
        f"- unique protein sequences:        **{proteins.height}**\n\n"
        "## Successful parses by ligand format\n\n"
        + fmt_counts.to_pandas().to_string(index=False) + "\n\n"
        "Parse failures are kept in `data/processed/pdbbind_complexes.parquet`\n"
        "with `ligand_parse_ok=false` / `protein_parse_ok=false` flags — they are\n"
        "NOT silently dropped.\n",
        encoding="utf-8")
    return complexes, ligands, proteins


def task_leakage_summaries(complexes: pl.DataFrame, ligands: pl.DataFrame) -> None:
    log_step("pre_step", "leakage_summaries")

    # Duplicate canonical SMILES (within PDBBind, one row per ligand with #complexes).
    dup_smiles = ligands.filter(pl.col("n_complexes") >= 2)
    dup_smiles.with_columns(
        pl.col("pdb_ids").list.join(",").alias("pdb_ids_csv")
    ).drop(["pdb_ids", "fp_bytes"]).write_csv(TABLES / "pdbbind_duplicate_ligands.csv")

    # Duplicate InChIKeys: same idea.
    dup_ik = (ligands.filter(pl.col("inchikey").is_not_null())
              .group_by("inchikey")
              .agg([pl.col("canonical_smiles").alias("canonical_smiles_list"),
                    pl.col("n_complexes").sum().alias("total_complexes")])
              .filter(pl.col("total_complexes") >= 2))
    dup_ik.with_columns(
        pl.col("canonical_smiles_list").list.join(" | ").alias("canonical_smiles_list")
    ).write_csv(TABLES / "pdbbind_duplicate_inchikeys.csv")

    # Scaffold frequencies.
    scaf_freq = (ligands.filter(pl.col("scaffold_smiles").is_not_null())
                 .group_by("scaffold_smiles")
                 .agg(pl.len().alias("n_unique_ligands"),
                      pl.col("n_complexes").sum().alias("n_complexes"))
                 .sort("n_complexes", descending=True))
    scaf_freq.write_csv(TABLES / "pdbbind_scaffold_frequencies.csv")

    # Ligand similarity edges at Tanimoto >= 0.8, capped to top-K nearest per
    # ligand. Pairwise on ~19K ligands is ~180M Tanimoto ops — manageable.
    sim_rows = _ligand_similarity_edges(ligands, threshold=0.8, top_k=5)
    sim_rows.write_csv(TABLES / "pdbbind_ligand_similarity_edges.csv")

    # Summary report.
    n_lig = ligands.height
    n_dup_smiles = dup_smiles.height
    n_dup_ik = dup_ik.height
    n_scaf = scaf_freq.height
    top_scaf = scaf_freq.head(10).to_pandas().to_string(index=False)
    (REPORTS / "pdbbind_leakage_summary.md").write_text(
        "# PDBBind chemical leakage summary\n\n"
        f"Generated by `vsleakkg.run_pdbbind` at {datetime.now(timezone.utc).isoformat()}.\n\n"
        f"- unique canonical SMILES:           **{n_lig}**\n"
        f"- canonical SMILES appearing in ≥ 2 complexes: **{n_dup_smiles}**\n"
        f"- InChIKeys appearing in ≥ 2 complexes:        **{n_dup_ik}**\n"
        f"- unique scaffolds:                  **{n_scaf}**\n"
        f"- ligand similarity edges (Tanimoto ≥ 0.8, top-5 per ligand): **{sim_rows.height}**\n\n"
        "## Top-10 most frequent scaffolds (across all complexes)\n\n"
        f"{top_scaf}\n\n"
        "Note: scaffold = generic Bemis-Murcko (RDKit `MurckoScaffold`). The\n"
        "empty scaffold (acyclic ligands) is reported as `\"\"` and tends to be\n"
        "the single most frequent row in this list.\n",
        encoding="utf-8")
    log_step("post_step", "leakage_summaries")


def _ligand_similarity_edges(ligands: pl.DataFrame, threshold: float, top_k: int) -> pl.DataFrame:
    """Pairwise Tanimoto >= threshold, top-K nearest neighbors per ligand."""
    sub = ligands.filter(pl.col("fp_bytes").is_not_null())
    fps = [vc.bytes_to_fp(b) for b in sub["fp_bytes"].to_list()]
    ids = sub["canonical_smiles"].to_list()
    n = len(fps)
    rows: List[tuple] = []
    log.info("PDBBind: computing pairwise Tanimoto for %d unique ligands", n)
    for i in range(n):
        if fps[i] is None:
            continue
        sims = vc.bulk_tanimoto(fps[i], fps)
        sims[i] = -1.0  # mask self
        hits = np.where(sims >= threshold)[0]
        if hits.size == 0:
            continue
        order = hits[np.argsort(-sims[hits])][:top_k]
        for j in order:
            rows.append((ids[i], ids[int(j)], float(sims[int(j)])))
        if (i + 1) % 2000 == 0:
            log.info("  similarity progress %d / %d  (%d edges so far)", i + 1, n, len(rows))
    return pl.DataFrame(rows, schema=["src_canonical_smiles", "dst_canonical_smiles", "tanimoto"],
                        orient="row")


# --------- graph build ---------

def task_build_graph(complexes: pl.DataFrame, ligands: pl.DataFrame,
                     proteins: pl.DataFrame, index_df: pl.DataFrame) -> None:
    log_step("pre_step", "build_graph")
    SRC = "PDBBind"
    nodes_rows: List[tuple] = []
    edges_rows: List[tuple] = []

    # Source + subset nodes.
    nodes_rows.append((f"src:{SRC}", "DatasetSource", SRC, "{}"))
    for subset in ("general", "refined", "core", "unknown"):
        nodes_rows.append((f"subset:{SRC}:{subset}", "PDBBindSubset", subset, "{}"))
    # The 2020R1 archive only ships the general PL index; refined/core are unknown.

    # Join index onto complexes by pdb_id (some complexes lack index rows; rare).
    full = complexes.join(index_df, on="pdb_id", how="left")

    # Ligand nodes (one per canonical SMILES). Re-use vb.ligand_id / scaffold_id.
    lig_id_for_smiles: Dict[str, str] = {}
    scaf_id_for_smiles: Dict[str, str] = {}
    for row in ligands.iter_rows(named=True):
        csmi = row["canonical_smiles"]
        if csmi is None:
            continue
        lid = vb.ligand_id(csmi)
        sid = vb.scaffold_id(row["scaffold_smiles"] or "")
        lig_id_for_smiles[csmi] = lid
        scaf_id_for_smiles[csmi] = sid
        nodes_rows.append((lid, "Ligand", csmi, json.dumps({
            "inchikey": row.get("inchikey"),
            "scaffold_smiles": row.get("scaffold_smiles") or "",
        })))
        nodes_rows.append((sid, "Scaffold", row.get("scaffold_smiles") or "",
                           json.dumps({"is_empty": row.get("scaffold_smiles") in (None, "")})))
        edges_rows.append((lid, sid, "ligand_has_scaffold", "{}"))

    # Protein nodes (one per exact-sequence cluster).
    prot_id_for_pdb: Dict[str, str] = {}
    for row in proteins.iter_rows(named=True):
        pid = f"prot:{SRC}:{row['seq_sha256'][:16]}"
        nodes_rows.append((pid, "Protein", row['seq_sha256'][:16],
                           json.dumps({"n_chains": row["n_chains"],
                                       "n_residues": row["n_residues"]})))
        for pdb in row["pdb_ids"]:
            prot_id_for_pdb[pdb] = pid

    # Complex nodes + per-complex edges.
    for row in full.iter_rows(named=True):
        pdb = row["pdb_id"]
        cid = f"complex:{SRC}:{pdb}"
        subset_id = f"subset:{SRC}:unknown"   # only general index is shipped
        nodes_rows.append((cid, "Complex", pdb, json.dumps({
            "year_bucket": row["year_bucket"],
            "release_year": row.get("release_year"),
            "resolution": row.get("resolution"),
            "is_nmr": row.get("is_nmr"),
        })))
        edges_rows.append((cid, f"src:{SRC}", "complex_from_source", "{}"))
        edges_rows.append((cid, subset_id, "complex_in_subset", "{}"))

        # Ligand link
        if row.get("ligand_parse_ok"):
            csmi = row["ligand_smiles_canonical"]
            lid = lig_id_for_smiles.get(csmi)
            if lid is not None:
                edges_rows.append((cid, lid, "complex_has_ligand", "{}"))

        # Protein link
        pid = prot_id_for_pdb.get(pdb)
        if pid is not None:
            edges_rows.append((cid, pid, "complex_has_protein", "{}"))

        # Pocket node + edge (one per complex; we keep them per-complex since
        # we don't compute pocket-similarity here).
        if row.get("has_pocket_file"):
            pkt = f"pocket:{SRC}:{pdb}"
            nodes_rows.append((pkt, "Pocket", pdb,
                               json.dumps({"n_atoms": row.get("pocket_n_atoms", 0)})))
            edges_rows.append((cid, pkt, "complex_has_pocket", "{}"))

        # Structure files
        for fld, kind in (("has_ligand_mol2", "ligand_mol2"),
                          ("has_ligand_sdf",  "ligand_sdf"),
                          ("has_pocket_file", "pocket_pdb"),
                          ("has_protein_pdb", "protein_pdb")):
            if row.get(fld):
                sf = f"struct:{SRC}:{pdb}:{kind}"
                nodes_rows.append((sf, "StructureFile", kind,
                                    json.dumps({"pdb_id": pdb})))
                edges_rows.append((cid, sf, "complex_has_structure_file", "{}"))

        # BindingMeasurement
        if row.get("affinity_parse_ok"):
            bm = f"bm:{SRC}:{pdb}"
            nodes_rows.append((bm, "BindingMeasurement",
                               f"{row['affinity_type']}{row['comparator']}{row['value']}{row['unit']}",
                               json.dumps({
                                   "value": row["value"], "unit": row["unit"],
                                   "value_M": row["value_M"], "p_value": row["p_value"],
                                   "p_name": row["p_name"], "comparator": row["comparator"],
                               })))
            edges_rows.append((cid, bm, "complex_has_binding_measurement", "{}"))
            at = f"affinity_type:{row['affinity_type']}"
            nodes_rows.append((at, "AffinityType", row["affinity_type"], "{}"))
            edges_rows.append((bm, at, "binding_measurement_has_type", "{}"))

    # ligand_similar_to_ligand edges.
    sim_csv = TABLES / "pdbbind_ligand_similarity_edges.csv"
    if sim_csv.exists():
        sim = pl.read_csv(sim_csv)
        for r in sim.iter_rows(named=True):
            a = lig_id_for_smiles.get(r["src_canonical_smiles"])
            b = lig_id_for_smiles.get(r["dst_canonical_smiles"])
            if a and b:
                edges_rows.append((a, b, "ligand_similar_to_ligand",
                                   json.dumps({"tanimoto": r["tanimoto"], "source": SRC})))

    nodes = (pl.DataFrame(nodes_rows, schema=["node_id", "node_type", "label", "props"], orient="row")
             .unique(subset=["node_id"]))
    edges = (pl.DataFrame(edges_rows, schema=["src", "dst", "edge_type", "props"], orient="row")
             .unique())
    nodes.write_parquet(PROCESSED / "pdbbind_nodes.parquet")
    edges.write_parquet(PROCESSED / "pdbbind_edges.parquet")
    (REPORTS / "pdbbind_graph_summary.md").write_text(_render_graph_summary("PDBBind", nodes, edges),
                                                       encoding="utf-8")
    log_step("post_step", f"build_graph nodes={nodes.height} edges={edges.height}")


def _render_graph_summary(label: str, nodes: pl.DataFrame, edges: pl.DataFrame) -> str:
    nbt = nodes.group_by("node_type").agg(pl.len().alias("n")).sort("node_type")
    eet = edges.group_by("edge_type").agg(pl.len().alias("n")).sort("edge_type")
    return (
        f"# {label} — graph summary\n\n"
        f"Nodes: **{nodes.height}** | Edges: **{edges.height}**\n\n"
        "## Nodes by type\n\n"
        + "\n".join(f"- {r['node_type']}: {r['n']}" for r in nbt.iter_rows(named=True))
        + "\n\n## Edges by type\n\n"
        + "\n".join(f"- {r['edge_type']}: {r['n']}" for r in eet.iter_rows(named=True))
        + "\n"
    )


# --------- cross-source merge ---------

def task_merge_with_mvp(complexes: pl.DataFrame, ligands: pl.DataFrame) -> None:
    log_step("pre_step", "merge_cross_source")
    mvp1_n = PROCESSED / "mvp1_nodes.parquet"
    mvp1_e = PROCESSED / "mvp1_edges.parquet"
    if not (mvp1_n.exists() and mvp1_e.exists()):
        log.warning("MVP-1 graph not found — skipping cross-source merge")
        return
    base_n = pl.read_parquet(mvp1_n)
    base_e = pl.read_parquet(mvp1_e)
    pdb_n  = pl.read_parquet(PROCESSED / "pdbbind_nodes.parquet")
    pdb_e  = pl.read_parquet(PROCESSED / "pdbbind_edges.parquet")
    nodes = pl.concat([base_n, pdb_n], how="vertical_relaxed").unique(subset=["node_id"])
    edges = pl.concat([base_e, pdb_e], how="vertical_relaxed").unique()

    # Same-InChIKey cross-source ligand links: in MVP-1 ligand nodes are keyed
    # by canonical SMILES, so a ligand that appears in BOTH PDBBind and (say)
    # DUD-E by canonical SMILES already shares a node — unique() collapses it
    # for free. The interesting case is "same InChIKey, different canonical
    # SMILES" (tautomer / stereo). We emit explicit `same_inchikey_as` edges
    # for those.
    pdb_lig_ik = {r["inchikey"]: vb.ligand_id(r["canonical_smiles"])
                  for r in ligands.iter_rows(named=True)
                  if r["inchikey"] and r["canonical_smiles"]}
    sources = []
    for src_label, parq, smi_col, ik_col in (
        ("LIT-PCBA", PROCESSED / "litpcba_ave_examples.parquet", "smiles_canonical", "inchikey"),
        ("DUD-E",    PROCESSED / "dude_examples.parquet",        "smiles_canonical", "inchikey"),
        ("DEKOIS",   PROCESSED / "dekois_examples.parquet",      "smiles_canonical", "inchikey"),
    ):
        if not parq.exists():
            continue
        df = (pl.scan_parquet(parq)
              .select([pl.col(smi_col).alias("canonical_smiles"),
                       pl.col(ik_col).alias("inchikey")])
              .filter(pl.col("inchikey").is_not_null() & pl.col("canonical_smiles").is_not_null())
              .unique(subset=["canonical_smiles"])
              .collect())
        sources.append((src_label, df))
        log.info("merge: %s unique ligands = %d", src_label, df.height)

    new_edges: List[tuple] = []
    for src_label, df in sources:
        for ik, smi in zip(df["inchikey"].to_list(), df["canonical_smiles"].to_list()):
            if ik not in pdb_lig_ik:
                continue
            pdb_lid = pdb_lig_ik[ik]
            other_lid = vb.ligand_id(smi)
            if pdb_lid == other_lid:
                continue   # same canonical SMILES already unifies them.
            new_edges.append((pdb_lid, other_lid, "same_inchikey_as",
                              json.dumps({"inchikey": ik, "other_source": src_label})))
    log.info("merge: %d new same_inchikey_as edges", len(new_edges))
    if new_edges:
        edges = pl.concat([
            edges,
            pl.DataFrame(new_edges, schema=["src", "dst", "edge_type", "props"], orient="row"),
        ], how="vertical_relaxed").unique()

    nodes.write_parquet(PROCESSED / "mvp1_plus_pdbbind_nodes.parquet")
    edges.write_parquet(PROCESSED / "mvp1_plus_pdbbind_edges.parquet")
    (REPORTS / "mvp1_plus_pdbbind_graph_summary.md").write_text(
        _render_graph_summary("MVP-1 + PDBBind combined", nodes, edges)
        + ("\n\n## Cross-source notes\n\n"
           "- Same-canonical-SMILES ligands are auto-deduped via `node_id = lig:md5(canonical_smiles)`.\n"
           "- `same_inchikey_as` edges link PDBBind ligands to other-source ligands\n"
           "  that share an InChIKey but differ in canonical SMILES (typically\n"
           "  tautomer or stereo encodings).\n"
           f"- {len(new_edges)} such edges emitted.\n"),
        encoding="utf-8")
    log_step("post_step", "merge_cross_source")


# --------- audit report ---------

def task_audit_report(complexes: pl.DataFrame, ligands: pl.DataFrame,
                      proteins: pl.DataFrame, index_df: pl.DataFrame) -> None:
    n = complexes.height
    n_lig_ok = int(complexes["ligand_parse_ok"].sum())
    n_prt_ok = int(complexes["protein_parse_ok"].sum())
    n_dup = int(ligands.filter(pl.col("n_complexes") >= 2).height)
    top_scaf = (ligands.filter(pl.col("scaffold_smiles").is_not_null())
                .group_by("scaffold_smiles")
                .agg(pl.col("n_complexes").sum().alias("n_complexes"))
                .sort("n_complexes", descending=True)
                .head(10).to_pandas().to_string(index=False))

    cross = ""
    csv_sim = TABLES / "pdbbind_ligand_similarity_edges.csv"
    sim_n = pl.read_csv(csv_sim).height if csv_sim.exists() else 0

    cross_overlap = _cross_source_overlap_table(ligands)
    cross_md = cross_overlap.to_pandas().to_string(index=False) if not cross_overlap.is_empty() else "(no MVP-1 sources to compare against)"

    body = (
        "# PDBBind audit report\n\n"
        f"Generated by `vsleakkg.run_pdbbind` at {datetime.now(timezone.utc).isoformat()}.\n\n"
        "## Detected version\n\n"
        "**PDBbind v2020R1 with re-processed structures from PDBbind v2024**\n"
        "(per `data/raw/PBDBind/extracted/index/README`, edited by Renxiao Wang,\n"
        "last update 2025-08-04). Only the protein-ligand (PL) general index is\n"
        "shipped in this archive.\n\n"
        "## Counts\n\n"
        f"- complexes in index:         **{index_df.height}**\n"
        f"- complexes parsed:           **{n}**\n"
        f"- ligand parses succeeded:    **{n_lig_ok}** ({n_lig_ok/n:.1%})\n"
        f"- protein parses succeeded:   **{n_prt_ok}** ({n_prt_ok/n:.1%})\n"
        f"- ligand parse failures:      **{n - n_lig_ok}**\n"
        f"- protein parse failures:     **{n - n_prt_ok}**\n"
        f"- subsets:                    only `general` ships in this archive\n"
        f"  (refined/core membership is not encoded — flagged as `unknown`).\n\n"
        "## Affinity summary\n\n"
        + index_df.group_by(["affinity_type", "unit"]).agg(pl.len().alias("n"))
            .sort("n", descending=True).head(20).to_pandas().to_string(index=False)
        + "\n\n"
        "## Chemistry leakage\n\n"
        f"- unique canonical SMILES:    **{ligands.height}**\n"
        f"- unique scaffolds:           "
        f"**{ligands.filter(pl.col('scaffold_smiles').is_not_null()).select('scaffold_smiles').n_unique()}**\n"
        f"- canonical SMILES in ≥ 2 complexes: **{n_dup}**\n"
        f"- ligand similarity edges at Tanimoto ≥ 0.8 (top-5/ligand): **{sim_n}**\n\n"
        "### Top-10 most frequent scaffolds (by complex count)\n\n"
        f"{top_scaf}\n\n"
        "## Cross-source overlap (InChIKey-matched ligands)\n\n"
        f"{cross_md}\n\n"
        "## Limitations\n\n"
        "- **No full protein sequence clustering**: we only deduplicate by\n"
        "  exact concatenated CA-residue sequence (sha256). Lightweight tools\n"
        "  like CD-HIT / MMseqs2 are NOT run here. PDBBind contains many\n"
        "  near-identical sequences (e.g., point mutants); these are NOT\n"
        "  collapsed in our graph.\n"
        "- **No pocket structural similarity**: pocket atoms are counted but\n"
        "  pocket geometry / pharmacophore similarity is deferred.\n"
        "- **No LP-PDBBind / CleanSplit comparison yet**: those use their own\n"
        "  curated splits. We can layer them on once the official files are\n"
        "  pulled.\n"
        "- **PDBBind is an affinity/complex dataset, not a VS retrieval\n"
        "  benchmark.** Active-decoy diagnostics from LIT-PCBA / DUD-E / DEKOIS\n"
        "  do not apply directly here. We only emit chemistry-level audit\n"
        "  features (duplicates, scaffolds, analog edges) and rely on the\n"
        "  combined graph to surface cross-source leakage.\n\n"
        "## Next steps\n\n"
        "1. Add CD-HIT / MMseqs2 protein sequence clustering at ≥ 30% identity\n"
        "   so 'protein-cluster leakage' becomes computable.\n"
        "2. Pull LP-PDBBind / CleanSplit splits and audit which complexes leak\n"
        "   across their train/test partitions by canonical SMILES, InChIKey,\n"
        "   scaffold, ECFP4 ≥ 0.8.\n"
        "3. Compute pocket-based features (pocket atom count, pocket-residue\n"
        "   composition) and a pocket similarity baseline.\n"
        "4. Cross-source: extend `same_inchikey_as` to a stronger\n"
        "   `same_scaffold_as` edge between PDBBind and DUD-E / DEKOIS so we\n"
        "   can audit how many DUD-E targets have their ligands sitting inside\n"
        "   a PDBBind crystallographic complex.\n"
    )
    (REPORTS / "pdbbind_audit_report.md").write_text(body, encoding="utf-8")


def _cross_source_overlap_table(ligands: pl.DataFrame) -> pl.DataFrame:
    rows = []
    pdb_iks = set(ligands.filter(pl.col("inchikey").is_not_null())["inchikey"].to_list())
    pdb_smis = set(ligands["canonical_smiles"].drop_nulls().to_list())
    for src_label, parq, smi_col, ik_col in (
        ("LIT-PCBA AVE", PROCESSED / "litpcba_ave_examples.parquet", "smiles_canonical", "inchikey"),
        ("DUD-E",        PROCESSED / "dude_examples.parquet",        "smiles_canonical", "inchikey"),
        ("DEKOIS",       PROCESSED / "dekois_examples.parquet",      "smiles_canonical", "inchikey"),
    ):
        if not parq.exists():
            continue
        df = (pl.scan_parquet(parq)
              .select([pl.col(smi_col).alias("canonical_smiles"),
                       pl.col(ik_col).alias("inchikey")])
              .filter(pl.col("inchikey").is_not_null())
              .unique(subset=["inchikey"])
              .collect())
        n_total = df.height
        n_ik_match = int(df["inchikey"].is_in(list(pdb_iks)).sum())
        n_smi_match = int(df["canonical_smiles"].is_in(list(pdb_smis)).sum())
        rows.append((src_label, n_total, n_smi_match, n_ik_match,
                     (n_ik_match / n_total) if n_total else 0.0))
    return pl.DataFrame(rows, schema=[
        "source", "n_unique_ligands", "shared_by_canonical_smiles",
        "shared_by_inchikey", "fraction_shared_by_inchikey",
    ], orient="row")


# --------- main ---------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--skip-merge", action="store_true")
    args = parser.parse_args(argv)

    log_step("pdbbind_start", "vs-leakkg")

    index_df = task_parse_index()
    complexes, ligands, proteins = task_parse_structures(args.workers)
    task_leakage_summaries(complexes, ligands)
    task_build_graph(complexes, ligands, proteins, index_df)
    if not args.skip_merge:
        task_merge_with_mvp(complexes, ligands)
    task_audit_report(complexes, ligands, proteins, index_df)

    log_step("pdbbind_end", "vs-leakkg")
    print()
    print("PDBBind/PBDBind processing complete. See:")
    print(" - outputs/reports/pdbbind_audit_report.md")
    print(" - outputs/reports/pdbbind_graph_summary.md")
    print(" - data/processed/pdbbind_nodes.parquet")
    print(" - data/processed/pdbbind_edges.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
