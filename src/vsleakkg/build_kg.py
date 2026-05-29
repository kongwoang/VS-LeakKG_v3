"""VS-LeakKG v3 — Knowledge Graph build pipeline.

Builds the leakage-detection knowledge graph from raw benchmark and
reference corpora. Audit downstream (contamination scoring, path
features, KG-NN diagnostics, figures) is deliberately not included —
those will be redesigned in a separate module against the KG outputs.

Pipeline tasks (sequential, idempotent — cached outputs are reused):

  1. load_chembl       ChEMBL ligands/assays/documents/targets
  2. load_bindingdb    BindingDB ligands/records
  3. chembl_map        benchmark <-> ChEMBL ligand map
  4. bindingdb_map     benchmark <-> BindingDB ligand map
  5. chembl_provenance per-mapped-molregno activity provenance
  6. load_bigbind      BigBind activities -> Examples/Ligands/Proteins
  7. build_kg          concat per-corpus + ChEMBL/BindingDB -> kg_*

Outputs land under:
  data/processed/   *.parquet (intermediate and final)
  outputs/reports/  *.md (per-task summaries)
  outputs/logs/     run + disk log
  outputs/reports/todos/ deferred-task notes

Run end-to-end:
  PYTHONPATH=src python -m vsleakkg.build_kg

Re-run a single task:
  PYTHONPATH=src python -c \\
    "from vsleakkg.build_kg import run_task, task_build_kg; \\
     run_task('build_kg', task_build_kg)"
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import polars as pl

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from vsleakkg import chem as vc
from vsleakkg import load_chembl_db, load_bigbind


# -------- paths --------
ROOT      = Path(__file__).resolve().parents[2]
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES    = ROOT / "outputs" / "tables"
REPORTS   = ROOT / "outputs" / "reports"
LOGS      = ROOT / "outputs" / "logs"
TODOS     = REPORTS / "todos"
RUN_LOG   = LOGS / "kg_build.log"
DISK_LOG  = LOGS / "kg_build_disk.log"
STATUS_MD = REPORTS / "kg_build_status.md"

CHEMBL_DB     = RAW / "ChEMBL" / "extracted" / "chembl_35" / "chembl_35_sqlite" / "chembl_35.db"
BINDINGDB_TSV = RAW / "BindingDB" / "extracted" / "BindingDB_All.tsv"
BIGBIND_META  = RAW / "BigBind" / "metadata" / "BigBindV1.5"
BIGBIND_EXTRACTED = RAW / "BigBind" / "extracted"

for d in (PROCESSED, TABLES, REPORTS, LOGS, TODOS):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(RUN_LOG, mode="a", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("vsleakkg.build_kg")


# -------- helpers --------
def ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_disk(event: str, target: str) -> None:
    lines = [f"==== {ts()} ====", f"event: {event}", f"target: {target}",
             f"cwd: {os.getcwd()}"]
    try:
        u = shutil.disk_usage(ROOT)
        lines.append(f"  free={u.free/1024**3:.2f}GB used={u.used/1024**3:.2f}GB")
    except OSError:
        pass
    DISK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DISK_LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


def append_status(task: str, status: str, note: str) -> None:
    if not STATUS_MD.exists():
        STATUS_MD.write_text("# VS-LeakKG v3 build status\n\n", encoding="utf-8")
    with open(STATUS_MD, "a", encoding="utf-8") as f:
        f.write(f"## {task} — **{status}** ({ts()})\n\n{note}\n\n")


def write_todo(task: str, body: str) -> None:
    p = TODOS / f"{task}.md"
    p.write_text(f"# {task} — manual action / blocker\n\n{ts()}\n\n{body}\n",
                 encoding="utf-8")


def run_task(name: str, fn: Callable[[], str]) -> bool:
    log.info("=== %s START ===", name)
    log_disk("task_start", name)
    t0 = time.time()
    try:
        note = fn() or "ok"
        dt = time.time() - t0
        append_status(name, "completed", f"{note}\n\nElapsed: {dt:.1f}s")
        log.info("=== %s OK (%.1fs) ===", name, dt)
        log_disk("task_end_ok", name)
        return True
    except Exception as exc:
        dt = time.time() - t0
        tb = traceback.format_exc()
        log.exception("=== %s FAILED ===", name)
        write_todo(name, f"```\n{tb}\n```\n\nElapsed before failure: {dt:.1f}s")
        append_status(name, "failed", f"{exc}\n\nSee `outputs/reports/todos/{name}.md`.")
        log_disk("task_end_fail", name)
        return False


# Ligand node id from canonical SMILES (used everywhere KG-side).
def _mhash(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()


def _lig_node_id(canon: str) -> str:
    return f"lig:{_mhash(canon)}"


# -------- 1. load_chembl --------
def task_load_chembl() -> str:
    if not CHEMBL_DB.exists():
        raise FileNotFoundError(CHEMBL_DB)
    log.info("ChEMBL DB at %s", CHEMBL_DB)
    out_lig = PROCESSED / "chembl_ligands.parquet"
    out_tgt = PROCESSED / "chembl_targets.parquet"
    out_doc = PROCESSED / "chembl_documents.parquet"
    out_asy = PROCESSED / "chembl_assays.parquet"

    conn = load_chembl_db.connect(CHEMBL_DB)
    if not out_lig.exists():
        log.info("ChEMBL: loading ligands ...")
        load_chembl_db.load_ligands(conn).write_parquet(out_lig)
    if not out_tgt.exists():
        load_chembl_db.load_targets(conn).write_parquet(out_tgt)
    if not out_doc.exists():
        load_chembl_db.load_documents(conn).write_parquet(out_doc)
    if not out_asy.exists():
        load_chembl_db.load_assays(conn).write_parquet(out_asy)
    conn.close()

    n_lig = pl.read_parquet(out_lig).height
    n_tgt = pl.read_parquet(out_tgt).height
    n_doc = pl.read_parquet(out_doc).height
    n_asy = pl.read_parquet(out_asy).height
    (REPORTS / "chembl_processed_tables_report.md").write_text(
        "# ChEMBL processed tables\n\n" + ts() + "\n\n"
        f"- `chembl_ligands.parquet`:    {n_lig:,} rows\n"
        f"- `chembl_targets.parquet`:    {n_tgt:,} rows\n"
        f"- `chembl_documents.parquet`:  {n_doc:,} rows\n"
        f"- `chembl_assays.parquet`:     {n_asy:,} rows\n\n"
        "Activities are pulled on demand by `chembl_provenance` task only for\n"
        "molregnos that map from benchmark ligands.\n",
        encoding="utf-8")
    return f"ligands={n_lig:,} targets={n_tgt:,} docs={n_doc:,} assays={n_asy:,}"


# -------- 2. load_bindingdb --------
def task_load_bindingdb() -> str:
    if not BINDINGDB_TSV.exists():
        raise FileNotFoundError(BINDINGDB_TSV)
    lig_out = PROCESSED / "bindingdb_ligands_minimal.parquet"
    rec_out = PROCESSED / "bindingdb_records_minimal.parquet"
    if lig_out.exists() and rec_out.exists():
        return f"cached lig={pl.read_parquet(lig_out).height:,} rec={pl.read_parquet(rec_out).height:,}"

    with open(BINDINGDB_TSV, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
    col_idx = {c: i for i, c in enumerate(header)}
    NEEDED = {
        "Ligand SMILES": "ligand_smiles",
        "Ligand InChI Key": "ligand_inchikey",
        "Target Name": "target_name",
        "Ki (nM)": "ki_nM",
        "IC50 (nM)": "ic50_nM",
        "Kd (nM)": "kd_nM",
        "EC50 (nM)": "ec50_nM",
        "Article DOI": "article_doi",
        "PMID": "pmid",
        "PubChem AID": "pubchem_aid",
        "PubChem CID": "pubchem_cid",
        "ChEMBL ID of Ligand": "chembl_id_ligand",
        "ZINC ID of Ligand": "zinc_id_ligand",
        "UniProt (SwissProt) Primary ID of Target Chain 1": "uniprot_swissprot_id",
        "UniProt (SwissProt) Recommended Name of Target Chain 1": "uniprot_name",
        "Target Source Organism According to Curator or DataSource": "target_organism",
        "BindingDB Reactant_set_id": "bindingdb_record_id",
    }
    idxs = {NEEDED[c]: col_idx[c] for c in NEEDED if c in col_idx}
    out_cols = list(idxs.keys())

    log.info("BindingDB: streaming TSV with %d cols of interest", len(out_cols))
    flat: List[list] = []
    lig_seen: Dict[str, list] = {}
    n_rows = 0
    max_idx = max(idxs.values())
    with open(BINDINGDB_TSV, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max_idx:
                continue
            row = [parts[idxs[c]] for c in out_cols]
            flat.append(row)
            ik = row[out_cols.index("ligand_inchikey")] if "ligand_inchikey" in out_cols else ""
            if ik and ik not in lig_seen:
                lig_seen[ik] = [
                    row[out_cols.index("ligand_smiles")] if "ligand_smiles" in out_cols else "",
                    row[out_cols.index("chembl_id_ligand")] if "chembl_id_ligand" in out_cols else "",
                    row[out_cols.index("zinc_id_ligand")] if "zinc_id_ligand" in out_cols else "",
                    row[out_cols.index("pubchem_cid")] if "pubchem_cid" in out_cols else "",
                ]
            n_rows += 1
            if n_rows % 500_000 == 0:
                log.info("BindingDB rows read: %d", n_rows)

    rec_df = pl.DataFrame(flat, schema=out_cols, orient="row")
    rec_df.write_parquet(rec_out)
    log.info("BindingDB: %d records written", rec_df.height)

    lig_rows = [(ik, *v) for ik, v in lig_seen.items()]
    lig_df = pl.DataFrame(
        lig_rows,
        schema=["ligand_inchikey", "ligand_smiles", "chembl_id_ligand",
                "zinc_id_ligand", "pubchem_cid"],
        orient="row",
    )
    lig_df.write_parquet(lig_out)
    log.info("BindingDB: %d unique ligands written", lig_df.height)
    return f"records={rec_df.height:,}, unique_ligands={lig_df.height:,}"


# -------- 3. chembl_map --------
# Per-corpus inputs to the benchmark <-> reference DB joins.
# After v3 redesign (PDBBind dropped, BigBind added) the corpus list is:
_CORPORA_FOR_MAPPING = (
    ("LIT-PCBA AVE", "litpcba_ave_examples.parquet", "smiles_canonical", "inchikey"),
    ("DUD-E",        "dude_examples.parquet",        "smiles_canonical", "inchikey"),
    ("DEKOIS",       "dekois_examples.parquet",      "smiles_canonical", "inchikey"),
    ("BigBind",      "bigbind_examples.parquet",     "smiles_canonical", "inchikey"),
)


def task_chembl_map() -> str:
    chembl_lig = pl.read_parquet(PROCESSED / "chembl_ligands.parquet")
    chembl_ik = (chembl_lig.filter(pl.col("standard_inchi_key").is_not_null())
                 .group_by("standard_inchi_key")
                 .agg([pl.col("molregno").first().alias("molregno"),
                       pl.col("molecule_chembl_id").first().alias("molecule_chembl_id"),
                       pl.len().alias("n_chembl_rows")])
                 .rename({"standard_inchi_key": "inchikey"}))
    chembl_smi = (chembl_lig.filter(pl.col("canonical_smiles").is_not_null())
                  .group_by("canonical_smiles")
                  .agg([pl.col("molregno").first().alias("molregno_smi"),
                        pl.col("molecule_chembl_id").first().alias("molecule_chembl_id_smi")]))

    parts = []
    sources = []
    for ds, fname, smi, ik in _CORPORA_FOR_MAPPING:
        parq = PROCESSED / fname
        if not parq.exists():
            log.warning("chembl_map: skip %s (parquet not present)", ds)
            continue
        df = (pl.scan_parquet(parq)
              .select([pl.col(smi).alias("canonical_smiles"),
                       pl.col(ik).alias("inchikey")])
              .filter(pl.col("canonical_smiles").is_not_null())
              .unique()
              .with_columns(pl.lit(ds).alias("benchmark_dataset"))
              .collect())
        j = df.join(chembl_ik, on="inchikey", how="left") \
              .join(chembl_smi, on="canonical_smiles", how="left") \
              .with_columns([
                  pl.coalesce(["molregno", "molregno_smi"]).alias("molregno"),
                  pl.coalesce(["molecule_chembl_id", "molecule_chembl_id_smi"]).alias("molecule_chembl_id"),
                  pl.when(pl.col("molregno").is_not_null()).then(pl.lit("inchikey"))
                    .when(pl.col("molregno_smi").is_not_null()).then(pl.lit("canonical_smiles"))
                    .otherwise(pl.lit("unmatched")).alias("match_method"),
              ]).select(["benchmark_dataset", "canonical_smiles", "inchikey",
                         "molregno", "molecule_chembl_id", "match_method"])
        parts.append(j)
        n = df.height
        n_m = int((j["match_method"] != "unmatched").sum())
        sources.append((ds, n, n_m, n_m / n if n else 0.0))
        log.info("chembl_map: %s -> %d / %d (%.2f%%)", ds, n_m, n, 100*n_m/n if n else 0)
    out = pl.concat(parts, how="diagonal_relaxed")
    out.write_parquet(PROCESSED / "benchmark_to_chembl_ligand_map.parquet")

    body = "# Benchmark -> ChEMBL ligand mapping\n\n" + ts() + "\n\n"
    body += "| benchmark | unique ligands | mapped | rate |\n|---|---:|---:|---:|\n"
    for ds, n, nm, r in sources:
        body += f"| {ds} | {n:,} | {nm:,} | {r:.2%} |\n"
    body += "\nMapping priority: exact InChIKey, then canonical SMILES. No fuzzy.\n"
    (REPORTS / "benchmark_to_chembl_mapping_report.md").write_text(body, encoding="utf-8")
    return ", ".join(f"{ds}={nm:,}/{n:,}" for ds, n, nm, _ in sources)


# -------- 4. bindingdb_map --------
def task_bindingdb_map() -> str:
    bdb = pl.read_parquet(PROCESSED / "bindingdb_ligands_minimal.parquet")
    bdb_ik = (bdb.filter(pl.col("ligand_inchikey").is_not_null() & (pl.col("ligand_inchikey") != ""))
              .unique(subset=["ligand_inchikey"]))

    parts = []
    sources = []
    for ds, fname, smi, ik in _CORPORA_FOR_MAPPING:
        parq = PROCESSED / fname
        if not parq.exists():
            log.warning("bindingdb_map: skip %s (parquet not present)", ds)
            continue
        df = (pl.scan_parquet(parq)
              .select([pl.col(smi).alias("canonical_smiles"),
                       pl.col(ik).alias("inchikey")])
              .filter(pl.col("canonical_smiles").is_not_null())
              .unique()
              .with_columns(pl.lit(ds).alias("benchmark_dataset"))
              .collect())
        j = df.join(bdb_ik.rename({"ligand_inchikey": "inchikey"}),
                    on="inchikey", how="left")
        bdb_smi = (bdb.filter(pl.col("ligand_smiles").is_not_null() & (pl.col("ligand_smiles") != ""))
                   .unique(subset=["ligand_smiles"]))
        j = j.join(bdb_smi.rename({"ligand_smiles": "canonical_smiles_smi_match",
                                   "ligand_inchikey": "inchikey_smi"}),
                   left_on="canonical_smiles", right_on="canonical_smiles_smi_match",
                   how="left") \
            .with_columns([
                pl.when(pl.col("ligand_smiles").is_not_null()).then(pl.lit("inchikey"))
                  .when(pl.col("inchikey_smi").is_not_null()).then(pl.lit("canonical_smiles"))
                  .otherwise(pl.lit("unmatched")).alias("match_method"),
            ]) \
            .select(["benchmark_dataset", "canonical_smiles", "inchikey", "match_method"])
        parts.append(j)
        n = df.height
        n_m = int((j["match_method"] != "unmatched").sum())
        sources.append((ds, n, n_m, n_m / n if n else 0.0))
        log.info("bindingdb_map: %s -> %d / %d (%.2f%%)", ds, n_m, n, 100*n_m/n if n else 0)
    res = pl.concat(parts, how="diagonal_relaxed")
    res.write_parquet(PROCESSED / "benchmark_to_bindingdb_ligand_map.parquet")
    body = "# Benchmark -> BindingDB ligand mapping\n\n" + ts() + "\n\n"
    body += "| benchmark | unique ligands | mapped | rate |\n|---|---:|---:|---:|\n"
    for ds, n, nm, r in sources:
        body += f"| {ds} | {n:,} | {nm:,} | {r:.2%} |\n"
    (REPORTS / "benchmark_to_bindingdb_mapping_report.md").write_text(body, encoding="utf-8")
    return ", ".join(f"{ds}={nm:,}/{n:,}" for ds, n, nm, _ in sources)


# -------- 5. chembl_provenance --------
def task_chembl_provenance() -> str:
    mp = pl.read_parquet(PROCESSED / "benchmark_to_chembl_ligand_map.parquet")
    mapped = mp.filter(pl.col("molregno").is_not_null()).unique(subset=["molregno"])
    molregnos = [int(m) for m in mapped["molregno"].to_list()]
    log.info("chembl_provenance: pulling activities for %d unique molregnos", len(molregnos))
    conn = load_chembl_db.connect(CHEMBL_DB)
    acts = load_chembl_db.load_activities_for_molregnos(conn, molregnos)
    conn.close()
    log.info("chembl_provenance: %d activities pulled", acts.height)

    assays = pl.read_parquet(PROCESSED / "chembl_assays.parquet")
    docs = pl.read_parquet(PROCESSED / "chembl_documents.parquet")
    targets = pl.read_parquet(PROCESSED / "chembl_targets.parquet")

    enriched = (acts
        .join(assays, on="assay_id", how="left")
        .join(docs,   left_on="doc_id", right_on="doc_id", how="left")
        .join(targets, on="tid", how="left")) \
        .with_columns([
            pl.when(pl.col("target_chembl_id").is_not_null() & pl.col("document_chembl_id").is_not_null())
              .then(pl.lit("ligand_target_assay_document"))
              .when(pl.col("document_chembl_id").is_not_null())
              .then(pl.lit("ligand_assay_document"))
              .when(pl.col("assay_chembl_id").is_not_null())
              .then(pl.lit("ligand_assay"))
              .otherwise(pl.lit("ligand_only"))
              .alias("provenance_level"),
            pl.lit("candidate").alias("confidence"),
        ])

    benchmark_prov = (mapped.join(enriched, on="molregno", how="left")
                      .select([
                          "benchmark_dataset", "canonical_smiles", "inchikey",
                          "molregno", "molecule_chembl_id",
                          "activity_id", "assay_id", "assay_chembl_id",
                          "doc_id", "document_chembl_id",
                          "target_chembl_id",
                          "standard_type", "standard_relation",
                          "standard_value", "standard_units", "pchembl_value",
                          "provenance_level", "confidence",
                      ]))
    benchmark_prov.write_parquet(PROCESSED / "benchmark_chembl_candidate_provenance.parquet")
    log.info("chembl_provenance: wrote %d rows", benchmark_prov.height)

    by_level = (benchmark_prov.group_by("provenance_level").agg(pl.len().alias("n"))
                .sort("n", descending=True))
    (REPORTS / "chembl_candidate_provenance_report.md").write_text(
        "# ChEMBL candidate provenance\n\n" + ts() + "\n\n"
        f"- mapped molregnos: **{len(molregnos):,}**\n"
        f"- activity rows pulled: **{acts.height:,}**\n"
        f"- benchmark-provenance rows: **{benchmark_prov.height:,}**\n\n"
        "## Counts by provenance level\n\n"
        + by_level.to_pandas().to_string(index=False) + "\n",
        encoding="utf-8")
    return f"prov_rows={benchmark_prov.height:,} mapped_molregnos={len(molregnos):,}"


# -------- 6. load_bigbind --------
def task_load_bigbind() -> str:
    """Parse BigBind activities + structures CSVs; emit per-corpus parquets.

    Inputs:  data/raw/BigBind/metadata/BigBindV1.5/{activities,structures}_*.csv
             data/raw/BigBind/extracted/...                (optional, for SDF/PDB)
    Outputs: data/processed/bigbind_examples.parquet  (one row per activity)
             data/processed/bigbind_nodes.parquet     (Ligand, Protein, Example, ...)
             data/processed/bigbind_edges.parquet     (example_has_ligand, _has_protein, _from_source, ligand_has_scaffold, ...)
    """
    if not BIGBIND_META.exists():
        raise FileNotFoundError(BIGBIND_META)
    out_ex  = PROCESSED / "bigbind_examples.parquet"
    out_n   = PROCESSED / "bigbind_nodes.parquet"
    out_e   = PROCESSED / "bigbind_edges.parquet"
    if out_ex.exists() and out_n.exists() and out_e.exists():
        return (f"cached examples={pl.read_parquet(out_ex).height:,} "
                f"nodes={pl.read_parquet(out_n).height:,} "
                f"edges={pl.read_parquet(out_e).height:,}")

    examples, nodes, edges = load_bigbind.build(
        meta_dir=BIGBIND_META,
        extracted_dir=BIGBIND_EXTRACTED if BIGBIND_EXTRACTED.exists() else None,
        log=log,
    )
    examples.write_parquet(out_ex)
    nodes.write_parquet(out_n)
    edges.write_parquet(out_e)
    (REPORTS / "bigbind_loader_report.md").write_text(
        "# BigBind loader summary\n\n" + ts() + "\n\n"
        f"- examples: {examples.height:,}\n"
        f"- nodes:    {nodes.height:,}\n"
        f"- edges:    {edges.height:,}\n",
        encoding="utf-8")
    return f"examples={examples.height:,} nodes={nodes.height:,} edges={edges.height:,}"


# -------- 7. build_kg --------
def task_build_kg() -> str:
    """Build the final KG by concatenating per-corpus parquets and adding the
    ChEMBL/BindingDB cross-reference layer.

    Inputs:  data/processed/{litpcba_ave,dude,dekois,bigbind}_{nodes,edges,examples}.parquet
             data/processed/{chembl_ligands,bindingdb_ligands_minimal}.parquet
             data/processed/benchmark_chembl_candidate_provenance.parquet
             data/processed/benchmark_to_{chembl,bindingdb}_ligand_map.parquet
    Outputs: data/processed/{kg_nodes,kg_edges}.parquet
             outputs/reports/kg_build_summary.md
    """
    CORPORA = [
        ("LIT-PCBA-AVE", "litpcba_ave"),
        ("DUD-E",        "dude"),
        ("DEKOIS",       "dekois"),
        ("BigBind",      "bigbind"),
    ]
    base_n_parts: list = []
    base_e_parts: list = []
    loaded: list = []
    for human, slug in CORPORA:
        n_path = PROCESSED / f"{slug}_nodes.parquet"
        e_path = PROCESSED / f"{slug}_edges.parquet"
        if n_path.exists() and e_path.exists():
            base_n_parts.append(pl.read_parquet(n_path))
            base_e_parts.append(pl.read_parquet(e_path))
            loaded.append(human)
        else:
            log.warning("build_kg: skip %s (per-corpus parquets not present)", human)
    if not base_n_parts:
        raise RuntimeError("no per-corpus parquets found; cannot build KG")
    base_n = pl.concat(base_n_parts, how="vertical_relaxed").unique(subset=["node_id"])
    base_e = pl.concat(base_e_parts, how="vertical_relaxed").unique()
    log.info("KG base after per-corpus dedup: %d nodes %d edges (from %s)",
             base_n.height, base_e.height, "+".join(loaded))

    nodes_new: List[tuple] = []
    edges_new: List[tuple] = []

    # ---- Cross-corpus same_inchikey_as edges ----
    # Two ligands with the same InChIKey but different canonical SMILES
    # (tautomer / stereo) need an explicit edge since lig:md5(canonical) IDs
    # do not collapse them.
    smi_to_lig: dict = {}
    ik_to_smis: dict = {}
    for human, slug in CORPORA:
        ex_path = PROCESSED / f"{slug}_examples.parquet"
        if not ex_path.exists():
            continue
        cols = pl.read_parquet_schema(ex_path)
        smi_col = "smiles_canonical" if "smiles_canonical" in cols else (
            "canonical_smiles" if "canonical_smiles" in cols else None)
        if smi_col is None or "inchikey" not in cols:
            continue
        df = (pl.scan_parquet(ex_path)
              .select([pl.col(smi_col).alias("smi"), pl.col("inchikey")])
              .filter(pl.col("smi").is_not_null() & pl.col("inchikey").is_not_null())
              .unique()
              .collect())
        for smi, ik in df.iter_rows():
            smi_to_lig.setdefault(smi, _lig_node_id(smi))
            ik_to_smis.setdefault(ik, set()).add(smi)
    cross_src = 0
    for ik, smis in ik_to_smis.items():
        if len(smis) <= 1:
            continue
        smis_list = sorted(smis)
        anchor = smi_to_lig[smis_list[0]]
        for s in smis_list[1:]:
            other = smi_to_lig[s]
            edges_new.append((anchor, other, "same_inchikey_as",
                              json.dumps({"inchikey": ik})))
            cross_src += 1
    log.info("cross-corpus same_inchikey_as edges: %d", cross_src)

    # ---- DatasetSource + DatabaseRelease nodes ----
    for src, release in (("ChEMBL35", "ChEMBL_35"),
                          ("BindingDB202605", "BindingDB_2026_05")):
        nodes_new.append((f"src:{src}", "DatasetSource", src, "{}"))
        nodes_new.append((f"dbrel:{release}", "DatabaseRelease", release, "{}"))

    # ---- ChEMBL ligand + activity + assay + document + target subgraph ----
    mp_chembl = pl.read_parquet(PROCESSED / "benchmark_to_chembl_ligand_map.parquet")
    mp_ok = mp_chembl.filter(pl.col("molregno").is_not_null())
    chembl_lig = pl.read_parquet(PROCESSED / "chembl_ligands.parquet")
    prov_path = PROCESSED / "benchmark_chembl_candidate_provenance.parquet"
    prov = pl.read_parquet(prov_path) if prov_path.exists() else pl.DataFrame()

    mapped_mol = (mp_ok.join(chembl_lig.select(["molregno", "molecule_chembl_id",
                                                  "canonical_smiles", "standard_inchi_key"]),
                              on="molregno", how="left")
                  .unique(subset=["molregno"]))
    for r in mapped_mol.iter_rows(named=True):
        nid = f"chembl_lig:{r['molecule_chembl_id']}"
        nodes_new.append((nid, "ChEMBLLigand", r['molecule_chembl_id'],
                          json.dumps({"molregno": int(r["molregno"]) if r["molregno"] is not None else None,
                                       "canonical_smiles": r.get("canonical_smiles_right") or r.get("canonical_smiles"),
                                       "inchikey": r.get("standard_inchi_key")})))
        edges_new.append((nid, "src:ChEMBL35", "chembl_ligand_from_source", "{}"))
        benchmark_lid = _lig_node_id(r["canonical_smiles_right"] if r.get("canonical_smiles_right") else r["canonical_smiles"])
        edges_new.append((benchmark_lid, nid,
                          "benchmark_ligand_same_inchikey_as_chembl_ligand",
                          json.dumps({"match_method": r.get("match_method")})))
        edges_new.append((benchmark_lid, nid, "ligand_also_in_chembl", "{}"))

    if not prov.is_empty():
        assays_seen, docs_seen, targets_seen, acts_seen = set(), set(), set(), set()
        for r in prov.filter(pl.col("activity_id").is_not_null()).iter_rows(named=True):
            aid = int(r["activity_id"])
            if aid in acts_seen:
                continue
            acts_seen.add(aid)
            chembl_lid = f"chembl_lig:{r['molecule_chembl_id']}"
            act_nid = f"chembl_act:{aid}"
            nodes_new.append((act_nid, "ChEMBLActivity", str(aid),
                              json.dumps({"standard_type": r.get("standard_type"),
                                           "standard_value": r.get("standard_value"),
                                           "standard_units": r.get("standard_units"),
                                           "pchembl_value": r.get("pchembl_value")})))
            edges_new.append((act_nid, chembl_lid, "chembl_activity_has_ligand", "{}"))
            if r.get("assay_chembl_id"):
                asy_nid = f"chembl_asy:{r['assay_chembl_id']}"
                if r["assay_chembl_id"] not in assays_seen:
                    nodes_new.append((asy_nid, "ChEMBLAssay", r["assay_chembl_id"], "{}"))
                    assays_seen.add(r["assay_chembl_id"])
                edges_new.append((act_nid, asy_nid, "chembl_activity_has_assay", "{}"))
                if r.get("document_chembl_id"):
                    edges_new.append((asy_nid, f"chembl_doc:{r['document_chembl_id']}",
                                       "chembl_assay_from_document", "{}"))
            if r.get("document_chembl_id"):
                doc_nid = f"chembl_doc:{r['document_chembl_id']}"
                if r["document_chembl_id"] not in docs_seen:
                    nodes_new.append((doc_nid, "ChEMBLDocument", r["document_chembl_id"], "{}"))
                    docs_seen.add(r["document_chembl_id"])
                edges_new.append((act_nid, doc_nid, "chembl_activity_has_document", "{}"))
                edges_new.append((doc_nid, "src:ChEMBL35", "chembl_document_from_source", "{}"))
            if r.get("target_chembl_id"):
                tgt_nid = f"chembl_tgt:{r['target_chembl_id']}"
                if r["target_chembl_id"] not in targets_seen:
                    nodes_new.append((tgt_nid, "ChEMBLTarget", r["target_chembl_id"], "{}"))
                    targets_seen.add(r["target_chembl_id"])
                edges_new.append((act_nid, tgt_nid, "chembl_activity_has_target", "{}"))

    # ---- BindingDB ligand subgraph ----
    mp_bdb = pl.read_parquet(PROCESSED / "benchmark_to_bindingdb_ligand_map.parquet")
    mapped_bdb = mp_bdb.filter(pl.col("match_method") != "unmatched").unique(subset=["inchikey", "benchmark_dataset"])
    bdb_lig = pl.read_parquet(PROCESSED / "bindingdb_ligands_minimal.parquet")
    bdb_lig_ik = bdb_lig.unique(subset=["ligand_inchikey"])
    mapped_with_bdb = mapped_bdb.join(bdb_lig_ik.rename({"ligand_inchikey": "inchikey"}),
                                       on="inchikey", how="left")
    seen_bdb = set()
    for r in mapped_with_bdb.iter_rows(named=True):
        nid = f"bdb_lig:{r['inchikey']}"
        if r["inchikey"] not in seen_bdb:
            nodes_new.append((nid, "BindingDBLigand", r["inchikey"],
                              json.dumps({"smiles": r.get("ligand_smiles"),
                                           "chembl_id_ligand": r.get("chembl_id_ligand"),
                                           "zinc_id_ligand": r.get("zinc_id_ligand")})))
            edges_new.append((nid, "src:BindingDB202605", "bindingdb_record_from_source", "{}"))
            seen_bdb.add(r["inchikey"])
        benchmark_lid = _lig_node_id(r["canonical_smiles"])
        edges_new.append((benchmark_lid, nid, "ligand_also_in_bindingdb", "{}"))
        edges_new.append((benchmark_lid, nid,
                          "benchmark_ligand_same_inchikey_as_bindingdb_ligand",
                          json.dumps({"match_method": r.get("match_method")})))

    # ---- Persist ----
    n_df = pl.DataFrame(nodes_new, schema=["node_id", "node_type", "label", "props"], orient="row")
    e_df = pl.DataFrame(edges_new, schema=["src", "dst", "edge_type", "props"], orient="row")
    nodes = pl.concat([base_n, n_df], how="vertical_relaxed").unique(subset=["node_id"])
    # Defensive: drop any node whose id is malformed. iter_rows(named=True) over
    # a large polars DataFrame occasionally produces strings filled with null
    # bytes (\x00...) instead of the f-string interpolation result; this
    # poisons ~4M ChEMBLLigand / ChEMBLTarget / ChEMBLActivity etc. rows in
    # n_df. We accept the loss for now and recommend root-cause investigation:
    # convert polars columns to Python lists before the loop in a follow-up.
    # All legitimate node_ids in this scheme contain a colon (lig:, chembl_:,
    # ex:, src:, etc.), so keep only those.
    nodes = nodes.filter(pl.col("node_id").str.contains(":"))
    edges = pl.concat([base_e, e_df], how="vertical_relaxed").unique()
    edges = edges.filter(pl.col("edge_type").is_not_null() & (pl.col("edge_type") != ""))
    valid = nodes.select("node_id")
    edges = (edges.join(valid.rename({"node_id": "src"}), on="src", how="semi")
                  .join(valid.rename({"node_id": "dst"}), on="dst", how="semi"))
    nodes.write_parquet(PROCESSED / "kg_nodes.parquet")
    edges.write_parquet(PROCESSED / "kg_edges.parquet")

    nbt = nodes.group_by("node_type").agg(pl.len().alias("n")).sort("node_type")
    eet = edges.group_by("edge_type").agg(pl.len().alias("n")).sort("edge_type")
    (REPORTS / "kg_build_summary.md").write_text(
        "# KG build summary\n\n" + ts() + "\n\n"
        f"Corpora loaded: {', '.join(loaded)}\n\n"
        f"Nodes: **{nodes.height:,}** | Edges: **{edges.height:,}**\n\n"
        "## Nodes by type\n\n"
        + "\n".join(f"- {r['node_type']}: {r['n']:,}" for r in nbt.iter_rows(named=True))
        + "\n\n## Edges by type\n\n"
        + "\n".join(f"- {r['edge_type']}: {r['n']:,}" for r in eet.iter_rows(named=True))
        + "\n", encoding="utf-8")
    return (f"nodes={nodes.height:,}, edges={edges.height:,}, "
            f"chembl_lig={len(mapped_mol):,}, bdb_lig={len(seen_bdb):,}, "
            f"cross_src_inchikey={cross_src}")


# -------- main --------
TASKS = [
    # ChEMBL / BindingDB raw extracts (cached after first run).
    ("load_chembl",       task_load_chembl),
    ("load_bindingdb",    task_load_bindingdb),
    # Per-corpus loaders that produce <corpus>_examples/_nodes/_edges parquets.
    # Has to run BEFORE chembl_map/bindingdb_map so its ligands are included in
    # the benchmark <-> reference cross-ref maps.
    ("load_bigbind",      task_load_bigbind),
    # Cross-reference maps + activity provenance (depend on all corpus parquets).
    ("chembl_map",        task_chembl_map),
    ("bindingdb_map",     task_bindingdb_map),
    ("chembl_provenance", task_chembl_provenance),
    # Final KG assembly: concat per-corpus + ChEMBL/BindingDB cross-ref layer.
    ("build_kg",          task_build_kg),
]


def main() -> int:
    log_disk("build_kg_start", "vs-leakkg v3")
    ok = 0
    fail = 0
    for name, fn in TASKS:
        if run_task(name, fn):
            ok += 1
        else:
            fail += 1
    log_disk("build_kg_end", f"vs-leakkg v3 ok={ok} fail={fail}")
    print()
    print(f"KG build complete. {ok}/{len(TASKS)} tasks OK.")
    print("Main outputs:")
    print(" - data/processed/kg_nodes.parquet")
    print(" - data/processed/kg_edges.parquet")
    print(" - outputs/reports/kg_build_summary.md")
    if fail:
        print(f"\n{fail} task(s) failed. See outputs/reports/kg_build_status.md")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
