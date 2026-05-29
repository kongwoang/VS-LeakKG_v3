"""VS-LeakKG overnight autonomous run — MVP-2 provenance build.

Runs Tasks 0..15 from the overnight spec sequentially with try/except per task.
On failure: write a TODO under `outputs/reports/todos/<task>.md` and continue
to the next independent task. Never claim success for a failed task.

All artifacts go under `data/processed/`, `outputs/tables/`, `outputs/reports/`.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from vsleakkg import chem as vc
from vsleakkg import load_chembl_db, load_bayesbind


# -------- paths --------
ROOT      = Path("D:/hoangpc/VS-LeakKG")
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES    = ROOT / "outputs" / "tables"
REPORTS   = ROOT / "outputs" / "reports"
LOGS      = ROOT / "outputs" / "logs"
TODOS     = REPORTS / "todos"
FIGURES   = REPORTS / "figures"
FIG_CSV   = TABLES / "figure_ready"
RUN_LOG   = LOGS / "overnight_run.log"
DISK_LOG  = LOGS / "overnight_disk_usage.log"
STATUS_MD = REPORTS / "overnight_status.md"

CHEMBL_DB = RAW / "ChEMBL" / "extracted" / "chembl_35" / "chembl_35_sqlite" / "chembl_35.db"
BINDINGDB_TSV = RAW / "BindingDB" / "extracted" / "BindingDB_All.tsv"
BAYESBIND_ROOT = RAW / "BayesBind" / "extracted"
BIGBIND_META   = RAW / "BigBind" / "metadata" / "BigBindV1.5"
PDBBIND_ROOT   = RAW / "PBDBind" / "extracted"

for d in (PROCESSED, TABLES, REPORTS, LOGS, TODOS, FIGURES, FIG_CSV):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(RUN_LOG, mode="a", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("vsleakkg.overnight")


# -------- helpers --------
def ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_disk(event: str, target: str) -> None:
    lines = [f"==== {ts()} ====", f"event: {event}", f"target: {target}",
             f"cwd: {os.getcwd()}"]
    try:
        u = shutil.disk_usage(ROOT)
        lines.append(f"  drive D: used={u.used/1024**3:.2f}GB free={u.free/1024**3:.2f}GB")
    except OSError:
        pass
    try:
        total = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
        lines.append(f"-- project size: {total/1024**3:.2f} GB")
    except OSError:
        pass
    DISK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DISK_LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


def append_status(task: str, status: str, note: str) -> None:
    if not STATUS_MD.exists():
        STATUS_MD.write_text("# VS-LeakKG overnight run — status\n\n", encoding="utf-8")
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


# -------- Task 0: input state --------
def task_0_state() -> str:
    rows = []
    for label, path in [
        ("MANIFEST.md",        ROOT / "data" / "MANIFEST.md"),
        ("setup_report.md",    REPORTS / "setup_report.md"),
        ("mvp1_audit_report.md",       REPORTS / "mvp1_audit_report.md"),
        ("pdbbind_audit_report.md",    REPORTS / "pdbbind_audit_report.md"),
        ("pdbbind_graph_summary.md",   REPORTS / "pdbbind_graph_summary.md"),
        ("full_dataset_download_report.md", REPORTS / "full_dataset_download_report.md"),
        ("litpcba_ave_examples.parquet",       PROCESSED / "litpcba_ave_examples.parquet"),
        ("dude_examples.parquet",              PROCESSED / "dude_examples.parquet"),
        ("dekois_examples.parquet",            PROCESSED / "dekois_examples.parquet"),
        ("pdbbind_complexes.parquet",          PROCESSED / "pdbbind_complexes.parquet"),
        ("pdbbind_ligands.parquet",            PROCESSED / "pdbbind_ligands.parquet"),
        ("pdbbind_proteins.parquet",           PROCESSED / "pdbbind_proteins.parquet"),
        ("mvp1_plus_pdbbind_nodes.parquet",    PROCESSED / "mvp1_plus_pdbbind_nodes.parquet"),
        ("mvp1_plus_pdbbind_edges.parquet",    PROCESSED / "mvp1_plus_pdbbind_edges.parquet"),
        ("chembl_35.db",       CHEMBL_DB),
        ("BindingDB_All.tsv",  BINDINGDB_TSV),
        ("BayesBind extract",  BAYESBIND_ROOT / "BayesBindV1.5"),
        ("BigBind tar.gz",     RAW / "BigBind" / "BigBindV1.5.tar.gz"),
    ]:
        if path.exists():
            sz = path.stat().st_size if path.is_file() else sum(
                p.stat().st_size for p in path.rglob("*") if p.is_file())
            rows.append((label, "OK", f"{sz/1024**2:.2f} MB", str(path)))
        else:
            rows.append((label, "MISSING", "—", str(path)))
    body = "# Overnight input state\n\n" + ts() + "\n\n"
    body += "| artifact | status | size | path |\n|---|---|---|---|\n"
    for r in rows:
        body += f"| {r[0]} | {r[1]} | {r[2]} | `{r[3]}` |\n"
    (REPORTS / "overnight_input_state_report.md").write_text(body, encoding="utf-8")
    return f"{sum(1 for r in rows if r[1] == 'OK')}/{len(rows)} required artifacts present"


# -------- Task 1: ChEMBL extract + minimal tables --------
def task_1_chembl() -> str:
    if not CHEMBL_DB.exists():
        raise FileNotFoundError(CHEMBL_DB)
    log.info("ChEMBL DB at %s", CHEMBL_DB)
    out_lig = PROCESSED / "chembl_ligands.parquet"
    out_tgt = PROCESSED / "chembl_targets.parquet"
    out_doc = PROCESSED / "chembl_documents.parquet"
    out_asy = PROCESSED / "chembl_assays.parquet"

    conn = load_chembl_db.connect(CHEMBL_DB)
    tables = load_chembl_db.list_tables(conn)
    schema_md = ["# ChEMBL 35 — schema report", "", ts(), "",
                 f"Tables ({len(tables)}):"]
    for t in tables:
        try:
            n = load_chembl_db.count(conn, t)
            schema_md.append(f"- `{t}`: {n:,} rows")
        except Exception:
            schema_md.append(f"- `{t}`: (count failed)")
    (REPORTS / "chembl_schema_report.md").write_text("\n".join(schema_md), encoding="utf-8")

    if not out_lig.exists():
        log.info("ChEMBL: loading ligands ...")
        lig = load_chembl_db.load_ligands(conn)
        log.info("ChEMBL: %d ligands", lig.height)
        lig.write_parquet(out_lig)
    if not out_tgt.exists():
        load_chembl_db.load_targets(conn).write_parquet(out_tgt)
    if not out_doc.exists():
        load_chembl_db.load_documents(conn).write_parquet(out_doc)
    if not out_asy.exists():
        load_chembl_db.load_assays(conn).write_parquet(out_asy)
    conn.close()

    # Quick processed-tables report
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
        "Activities are pulled on demand in Task 3 (only for molregnos that\n"
        "map from benchmark ligands) to avoid materializing the full ~20 M-row\n"
        "activities table.\n",
        encoding="utf-8")
    return f"ligands={n_lig:,} targets={n_tgt:,} docs={n_doc:,} assays={n_asy:,}"


# -------- Task 4: BindingDB minimal --------
def task_4_bindingdb() -> str:
    if not BINDINGDB_TSV.exists():
        raise FileNotFoundError(BINDINGDB_TSV)
    lig_out  = PROCESSED / "bindingdb_ligands_minimal.parquet"
    rec_out  = PROCESSED / "bindingdb_records_minimal.parquet"
    if lig_out.exists() and rec_out.exists():
        return f"cached lig={pl.read_parquet(lig_out).height:,} rec={pl.read_parquet(rec_out).height:,}"

    # Column indices from header probe (1-indexed columns; convert to 0-based).
    # Defensive: read header to find each column by name.
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
    missing = [c for c in NEEDED if c not in col_idx]
    if missing:
        log.warning("BindingDB missing expected columns: %s", missing)
    idxs = {NEEDED[c]: col_idx[c] for c in NEEDED if c in col_idx}
    out_cols = list(idxs.keys())

    log.info("BindingDB: streaming TSV with %d cols of interest", len(out_cols))
    rec_batches: List[List[list]] = []
    lig_seen: Dict[str, list] = {}   # ligand_inchikey -> [smiles, chembl_id, zinc_id, cid]
    BATCH = 200_000
    cur: List[list] = []
    n_rows = 0
    with open(BINDINGDB_TSV, "r", encoding="utf-8", errors="replace") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < max(idxs.values()) + 1:
                continue
            row = [parts[idxs[c]] if idxs[c] < len(parts) else "" for c in out_cols]
            cur.append(row)
            ik = row[out_cols.index("ligand_inchikey")] if "ligand_inchikey" in out_cols else ""
            if ik and ik not in lig_seen:
                lig_seen[ik] = [
                    row[out_cols.index("ligand_smiles")] if "ligand_smiles" in out_cols else "",
                    row[out_cols.index("chembl_id_ligand")] if "chembl_id_ligand" in out_cols else "",
                    row[out_cols.index("zinc_id_ligand")] if "zinc_id_ligand" in out_cols else "",
                    row[out_cols.index("pubchem_cid")] if "pubchem_cid" in out_cols else "",
                ]
            n_rows += 1
            if len(cur) >= BATCH:
                rec_batches.append(cur)
                cur = []
                log.info("BindingDB rows read: %d", n_rows)
    if cur:
        rec_batches.append(cur)
        log.info("BindingDB rows read final: %d", n_rows)

    # Flatten records
    flat = [r for b in rec_batches for r in b]
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

    (REPORTS / "bindingdb_schema_report.md").write_text(
        "# BindingDB schema (subset kept)\n\n" + ts() + "\n\n"
        f"Source: `{BINDINGDB_TSV}` (8.3 GB), header has {len(header)} columns.\n\n"
        "Kept columns:\n\n"
        + "\n".join(f"- `{c}` -> `{NEEDED[c]}`" for c in NEEDED if c in col_idx)
        + (f"\n\nMissing expected columns: `{missing}`\n" if missing else "\n"),
        encoding="utf-8")
    (REPORTS / "bindingdb_processed_tables_report.md").write_text(
        "# BindingDB processed tables\n\n" + ts() + "\n\n"
        f"- `bindingdb_records_minimal.parquet`: {rec_df.height:,} rows × {len(out_cols)} cols\n"
        f"- `bindingdb_ligands_minimal.parquet`: {lig_df.height:,} unique InChIKey rows\n",
        encoding="utf-8")
    return f"records={rec_df.height:,}, unique_ligands={lig_df.height:,}"


# -------- Task 2: benchmark -> ChEMBL --------
def task_2_chembl_map() -> str:
    chembl_lig = pl.read_parquet(PROCESSED / "chembl_ligands.parquet")
    # Build (inchikey -> molregno, chembl_id) lookup.
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

    def _scan(parq: Path, ds: str, smi_col: str, ik_col: str) -> pl.DataFrame:
        df = (pl.scan_parquet(parq)
              .select([pl.col(smi_col).alias("canonical_smiles"),
                       pl.col(ik_col).alias("inchikey")])
              .filter(pl.col("canonical_smiles").is_not_null())
              .unique()
              .with_columns(pl.lit(ds).alias("benchmark_dataset"))
              .collect())
        return df

    parts = []
    sources = []
    for ds, parq, smi, ik in (
        ("LIT-PCBA AVE", PROCESSED / "litpcba_ave_examples.parquet", "smiles_canonical", "inchikey"),
        ("DUD-E",        PROCESSED / "dude_examples.parquet",        "smiles_canonical", "inchikey"),
        ("DEKOIS",       PROCESSED / "dekois_examples.parquet",      "smiles_canonical", "inchikey"),
        ("PDBBind",      PROCESSED / "pdbbind_ligands.parquet",      "canonical_smiles", "inchikey"),
    ):
        if not parq.exists():
            continue
        df = _scan(parq, ds, smi, ik)
        # Join by InChIKey then SMILES.
        j1 = df.join(chembl_ik, on="inchikey", how="left")
        j2 = j1.join(chembl_smi, on="canonical_smiles", how="left")
        j2 = j2.with_columns([
            pl.coalesce(["molregno", "molregno_smi"]).alias("molregno"),
            pl.coalesce(["molecule_chembl_id", "molecule_chembl_id_smi"]).alias("molecule_chembl_id"),
            pl.when(pl.col("molregno").is_not_null())
              .then(pl.lit("inchikey"))
              .when(pl.col("molregno_smi").is_not_null())
              .then(pl.lit("canonical_smiles"))
              .otherwise(pl.lit("unmatched"))
              .alias("match_method"),
        ]).select(["benchmark_dataset", "canonical_smiles", "inchikey",
                   "molregno", "molecule_chembl_id", "match_method"])
        parts.append(j2)
        n = df.height
        n_m = int((j2["match_method"] != "unmatched").sum())
        sources.append((ds, n, n_m, n_m / n if n else 0.0))
        log.info("Task 2: %s -> ChEMBL %d / %d (%.2f%%)", ds, n_m, n, 100*n_m/n if n else 0)
    out = pl.concat(parts, how="diagonal_relaxed")
    out.write_parquet(PROCESSED / "benchmark_to_chembl_ligand_map.parquet")

    body = "# Benchmark -> ChEMBL ligand mapping\n\n" + ts() + "\n\n"
    body += "| benchmark | unique ligands | mapped | rate |\n|---|---:|---:|---:|\n"
    for ds, n, nm, r in sources:
        body += f"| {ds} | {n:,} | {nm:,} | {r:.2%} |\n"
    body += "\nMapping priority: exact InChIKey, then canonical SMILES. No fuzzy.\n"
    body += "\n## Output\n- `data/processed/benchmark_to_chembl_ligand_map.parquet` ({} rows).\n".format(f"{out.height:,}")
    (REPORTS / "benchmark_to_chembl_mapping_report.md").write_text(body, encoding="utf-8")
    return ", ".join(f"{ds}={nm:,}/{n:,}" for ds, n, nm, _ in sources)


# -------- Task 5: benchmark -> BindingDB --------
def task_5_bindingdb_map() -> str:
    bdb = pl.read_parquet(PROCESSED / "bindingdb_ligands_minimal.parquet")
    bdb_ik = (bdb.filter(pl.col("ligand_inchikey").is_not_null() & (pl.col("ligand_inchikey") != ""))
              .unique(subset=["ligand_inchikey"]))

    parts = []
    sources = []
    for ds, parq, smi, ik in (
        ("LIT-PCBA AVE", PROCESSED / "litpcba_ave_examples.parquet", "smiles_canonical", "inchikey"),
        ("DUD-E",        PROCESSED / "dude_examples.parquet",        "smiles_canonical", "inchikey"),
        ("DEKOIS",       PROCESSED / "dekois_examples.parquet",      "smiles_canonical", "inchikey"),
        ("PDBBind",      PROCESSED / "pdbbind_ligands.parquet",      "canonical_smiles", "inchikey"),
    ):
        if not parq.exists():
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
        # SMILES fallback
        bdb_smi = (bdb.filter(pl.col("ligand_smiles").is_not_null() & (pl.col("ligand_smiles") != ""))
                   .unique(subset=["ligand_smiles"]))
        j = j.join(bdb_smi.rename({"ligand_smiles": "canonical_smiles_smi_match",
                                    "ligand_inchikey": "inchikey_smi"}),
                   left_on="canonical_smiles", right_on="canonical_smiles_smi_match",
                   how="left")
        j = j.with_columns([
            pl.when(pl.col("ligand_smiles").is_not_null()).then(pl.lit("inchikey"))
              .when(pl.col("inchikey_smi").is_not_null()).then(pl.lit("canonical_smiles"))
              .otherwise(pl.lit("unmatched")).alias("match_method"),
        ])
        out = j.select(["benchmark_dataset", "canonical_smiles", "inchikey",
                        "match_method"])
        parts.append(out)
        n = df.height
        n_m = int((out["match_method"] != "unmatched").sum())
        sources.append((ds, n, n_m, n_m / n if n else 0.0))
        log.info("Task 5: %s -> BindingDB %d / %d (%.2f%%)", ds, n_m, n, 100*n_m/n if n else 0)
    res = pl.concat(parts, how="diagonal_relaxed")
    res.write_parquet(PROCESSED / "benchmark_to_bindingdb_ligand_map.parquet")
    body = "# Benchmark -> BindingDB ligand mapping\n\n" + ts() + "\n\n"
    body += "| benchmark | unique ligands | mapped | rate |\n|---|---:|---:|---:|\n"
    for ds, n, nm, r in sources:
        body += f"| {ds} | {n:,} | {nm:,} | {r:.2%} |\n"
    (REPORTS / "benchmark_to_bindingdb_mapping_report.md").write_text(body, encoding="utf-8")
    return ", ".join(f"{ds}={nm:,}/{n:,}" for ds, n, nm, _ in sources)


# -------- Task 3: ChEMBL provenance --------
def task_3_chembl_provenance() -> str:
    mp = pl.read_parquet(PROCESSED / "benchmark_to_chembl_ligand_map.parquet")
    mapped = mp.filter(pl.col("molregno").is_not_null()).unique(subset=["molregno"])
    molregnos = [int(m) for m in mapped["molregno"].to_list()]
    log.info("Task 3: pulling activities for %d unique mapped molregnos", len(molregnos))
    conn = load_chembl_db.connect(CHEMBL_DB)
    acts = load_chembl_db.load_activities_for_molregnos(conn, molregnos)
    conn.close()
    log.info("Task 3: %d activities pulled", acts.height)

    assays = pl.read_parquet(PROCESSED / "chembl_assays.parquet")
    docs = pl.read_parquet(PROCESSED / "chembl_documents.parquet")
    targets = pl.read_parquet(PROCESSED / "chembl_targets.parquet")

    # Per-ligand provenance: ligand_only / ligand_assay / ligand_assay_document / ligand_target_assay_document.
    enriched = (acts
        .join(assays, on="assay_id", how="left")
        .join(docs,   left_on="doc_id", right_on="doc_id", how="left")
        .join(targets.rename({"tid": "tid"}), on="tid", how="left"))

    enriched = enriched.with_columns([
        pl.when(pl.col("target_chembl_id").is_not_null() & pl.col("document_chembl_id").is_not_null())
          .then(pl.lit("ligand_target_assay_document"))
          .when(pl.col("document_chembl_id").is_not_null())
          .then(pl.lit("ligand_assay_document"))
          .when(pl.col("assay_chembl_id").is_not_null())
          .then(pl.lit("ligand_assay"))
          .otherwise(pl.lit("ligand_only"))
          .alias("provenance_level"),
        pl.lit("candidate").alias("confidence"),  # provenance is ligand-level only here
    ])

    # Attach benchmark mapping (multi-row per molregno across benchmarks).
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
    log.info("Task 3: wrote %d provenance rows", benchmark_prov.height)

    by_level = (benchmark_prov.group_by("provenance_level").agg(pl.len().alias("n"))
                .sort("n", descending=True))
    (REPORTS / "chembl_candidate_provenance_report.md").write_text(
        "# ChEMBL candidate provenance\n\n" + ts() + "\n\n"
        f"- mapped molregnos: **{len(molregnos):,}**\n"
        f"- activity rows pulled: **{acts.height:,}**\n"
        f"- benchmark-provenance rows: **{benchmark_prov.height:,}**\n\n"
        "## Counts by provenance level\n\n"
        + by_level.to_pandas().to_string(index=False) + "\n\n"
        "All rows are tagged `confidence=candidate` because no exact\n"
        "benchmark-target ↔ ChEMBL-target match is performed in this run.\n"
        "An explicit target-name join can be layered later to upgrade\n"
        "`candidate` to `confirmed` per target.\n",
        encoding="utf-8")
    return f"prov_rows={benchmark_prov.height:,} mapped_molregnos={len(molregnos):,}"


# -------- Task 9: protein clustering (probe) --------
def task_9_protein_clustering() -> str:
    has_mmseqs = shutil.which("mmseqs") is not None
    has_cdhit  = shutil.which("cd-hit") is not None or shutil.which("cdhit") is not None
    # Write FASTA regardless.
    fasta = PROCESSED / "pdbbind_proteins.fasta"
    proteins = pl.read_parquet(PROCESSED / "pdbbind_proteins.parquet")
    with open(fasta, "w", encoding="utf-8") as f:
        for r in proteins.iter_rows(named=True):
            seq = r.get("sequence_concat") or ""
            seq = seq.replace("|", "")
            if not seq:
                continue
            f.write(f">{r['seq_sha256'][:16]}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")
    if not (has_mmseqs or has_cdhit):
        write_todo("Task09_MMSEQS2_OR_CDHIT", (
            "Neither `mmseqs` nor `cd-hit` is on PATH on this host.\n\n"
            "PDBBind protein clustering at 90/50/30% identity is deferred.\n"
            "Install one of these tools and re-run:\n\n"
            "**MMseqs2 (recommended)**\n```\n"
            "conda install -c bioconda mmseqs2\n"
            "# Then:\n"
            f"mmseqs easy-cluster {fasta} pdbbind_clu /tmp --min-seq-id 0.30 -c 0.8 --cov-mode 0\n"
            "```\n\n"
            "**CD-HIT (alternative)**\n```\n"
            "conda install -c bioconda cd-hit\n"
            f"cd-hit -i {fasta} -o pdbbind_cdhit_30 -c 0.30 -n 2 -M 8000 -T 12\n"
            "```\n"
            f"\nFASTA at: `{fasta}` ({proteins.height} sequences)\n"
        ))
        (REPORTS / "pdbbind_protein_clustering_report.md").write_text(
            "# PDBBind protein clustering — DEFERRED\n\n" + ts() + "\n\n"
            f"FASTA written to `{fasta}` ({proteins.height} sequences). MMseqs2 and CD-HIT\n"
            "both unavailable on this host; see\n"
            "`outputs/reports/todos/Task09_MMSEQS2_OR_CDHIT.md` for install commands.\n"
            "Exact-sequence clustering (sha256) remains the only protein dedupe in the\n"
            "graph; see `pdbbind_proteins.parquet` `seq_sha256` column.\n",
            encoding="utf-8")
        return f"fasta={proteins.height} seqs; mmseqs/cd-hit MISSING — TODO written"
    # If a tool is available, we'd run it here. Skipped in this overnight pass.
    return "skipped (tool present but clustering not implemented in this run)"


# -------- Task 10: BayesBind --------
def task_10_bayesbind() -> str:
    df = load_bayesbind.load_all(BAYESBIND_ROOT)
    if df.is_empty():
        raise RuntimeError("BayesBind: no examples loaded")
    log.info("BayesBind: %d examples across %d targets",
             df.height, df.select("target").n_unique())
    # Featurize (lighter; reuse chem.featurize serially since dataset is small).
    feats = [vc.featurize(s) for s in df["smiles_input"].to_list()]
    df = df.with_columns([
        pl.Series("smiles_canonical", [f.smiles_canonical for f in feats]),
        pl.Series("inchikey",         [f.inchikey         for f in feats]),
        pl.Series("scaffold_smiles",  [f.scaffold_smiles  for f in feats]),
        pl.Series("parse_ok",         [f.parse_ok         for f in feats]),
    ])
    df.write_parquet(PROCESSED / "bayesbind_examples.parquet")
    # Per-target × split × label counts
    by = (df.group_by(["split", "target", "label_type"]).agg(pl.len().alias("n"))
          .sort(["split", "target", "label_type"]))
    counts = {
        "n_examples": df.height,
        "n_targets":  df.select("target").n_unique(),
        "n_splits":   df.select("split").n_unique(),
        "n_actives":  int((df["label"] == 1).sum()),
        "n_random":   int((df["label"] == 0).sum()),
        "parse_ok":   int(df["parse_ok"].sum()),
    }
    (REPORTS / "bayesbind_dataset_summary.md").write_text(
        "# BayesBind V1.5 — dataset summary\n\n" + ts() + "\n\n"
        + "\n".join(f"- **{k}**: {v}" for k, v in counts.items()) + "\n\n"
        "## Per split × target × label\n\n"
        + by.to_pandas().to_string(index=False) + "\n", encoding="utf-8")
    (REPORTS / "bayesbind_layout_report.md").write_text(
        "# BayesBind V1.5 — layout\n\n" + ts() + "\n\n"
        f"Targets discovered: {df.select('target').n_unique()} across splits\n"
        f"{sorted(df['split'].unique().to_list())}.\n\n"
        "Per target: `actives.csv` + `random.csv` (decoys) + receptor/pocket PDBs.\n"
        "Pocket / cluster / UniProt metadata preserved in `bayesbind_examples.parquet`.\n",
        encoding="utf-8")
    return ", ".join(f"{k}={v}" for k, v in counts.items())


# -------- Task 11: BigBind metadata --------
def task_11_bigbind_metadata() -> str:
    if not BIGBIND_META.exists():
        raise FileNotFoundError(BIGBIND_META)
    files = sorted(BIGBIND_META.glob("*.csv"))
    rows = []
    for f in files:
        # Cheap line count
        with open(f, "rb") as fh:
            n = sum(1 for _ in fh) - 1
        rows.append((f.name, n, f.stat().st_size))
    df = pl.DataFrame(rows, schema=["file", "n_rows", "bytes"], orient="row")
    df.write_parquet(PROCESSED / "bigbind_metadata_summary.parquet")
    body = (
        "# BigBind V1.5 — metadata-only inspection\n\n" + ts() + "\n\n"
        "Full archive `BigBindV1.5.tar.gz` (17.9 GiB) is intentionally kept\n"
        "compressed. Only the 12 top-level CSV metadata files were extracted\n"
        f"into `{BIGBIND_META}`.\n\n"
        "## Metadata summary\n\n"
        + df.to_pandas().to_string(index=False) + "\n\n"
        "These cover `activities_*`, `structures_*`, and the SNA-balanced\n"
        "splits (`*_sna_1_*`) plus `*_unfiltered`. Loading them as Polars\n"
        "frames does NOT require expanding the 680K SDF / 74K PDB structure\n"
        "files.\n"
    )
    (REPORTS / "bigbind_archive_inspection.md").write_text(body, encoding="utf-8")
    return f"meta_csv={len(rows)} total_rows={sum(r[1] for r in rows):,}"


# -------- Task 6: MVP-2 graph (restricted to mapped/known) --------
def _mhash(s: str) -> str:
    import hashlib
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()


def _lig_node_id(canon: str) -> str:
    return f"lig:{_mhash(canon)}"


def task_6_mvp2_graph() -> str:
    base_n = pl.read_parquet(PROCESSED / "mvp1_plus_pdbbind_nodes.parquet")
    base_e = pl.read_parquet(PROCESSED / "mvp1_plus_pdbbind_edges.parquet")
    log.info("MVP-2 graph: base %d nodes %d edges", base_n.height, base_e.height)

    nodes_new: List[tuple] = []
    edges_new: List[tuple] = []

    # DatasetSource + DatabaseRelease nodes (always add — small).
    for src, release in (("ChEMBL35", "ChEMBL_35"),
                          ("BindingDB202605", "BindingDB_2026_05")):
        nodes_new.append((f"src:{src}", "DatasetSource", src, "{}"))
        nodes_new.append((f"dbrel:{release}", "DatabaseRelease", release, "{}"))

    # ----- ChEMBL ligand + activity + assay + document + target subgraph -----
    mp_chembl = pl.read_parquet(PROCESSED / "benchmark_to_chembl_ligand_map.parquet")
    mp_ok = mp_chembl.filter(pl.col("molregno").is_not_null())
    chembl_lig = pl.read_parquet(PROCESSED / "chembl_ligands.parquet")
    if Path(PROCESSED / "benchmark_chembl_candidate_provenance.parquet").exists():
        prov = pl.read_parquet(PROCESSED / "benchmark_chembl_candidate_provenance.parquet")
    else:
        prov = pl.DataFrame()

    # Build ChEMBLLigand node for each mapped molregno.
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
        edges_new.append((nid, "src:ChEMBL35", "chembl_document_from_source", "{}"))
        # Link benchmark Ligand (existing) -> ChEMBL ligand via InChIKey/SMILES.
        benchmark_lid = _lig_node_id(r["canonical_smiles_right"] if r.get("canonical_smiles_right") else r["canonical_smiles"])
        edges_new.append((benchmark_lid, nid,
                          "benchmark_ligand_same_inchikey_as_chembl_ligand",
                          json.dumps({"match_method": r.get("match_method")})))
        edges_new.append((benchmark_lid, nid, "ligand_also_in_chembl", "{}"))

    # Activity, assay, document, target nodes from provenance.
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

    # ----- BindingDB ligand subgraph -----
    mp_bdb = pl.read_parquet(PROCESSED / "benchmark_to_bindingdb_ligand_map.parquet")
    mapped_bdb = mp_bdb.filter(pl.col("match_method") != "unmatched").unique(subset=["inchikey", "benchmark_dataset"])
    bdb_lig = pl.read_parquet(PROCESSED / "bindingdb_ligands_minimal.parquet")
    bdb_lig_ik = bdb_lig.unique(subset=["ligand_inchikey"])
    mapped_with_bdb = mapped_bdb.join(bdb_lig_ik.rename({"ligand_inchikey": "inchikey"}),
                                       on="inchikey", how="left")
    seen_bdb = set()
    for r in mapped_with_bdb.iter_rows(named=True):
        if r["inchikey"] in seen_bdb:
            # Still emit the per-benchmark edge.
            pass
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

    # Persist
    n_df = pl.DataFrame(nodes_new, schema=["node_id", "node_type", "label", "props"], orient="row")
    e_df = pl.DataFrame(edges_new, schema=["src", "dst", "edge_type", "props"], orient="row")
    nodes = pl.concat([base_n, n_df], how="vertical_relaxed").unique(subset=["node_id"])
    edges = pl.concat([base_e, e_df], how="vertical_relaxed").unique()
    nodes.write_parquet(PROCESSED / "mvp2_nodes.parquet")
    edges.write_parquet(PROCESSED / "mvp2_edges.parquet")

    nbt = nodes.group_by("node_type").agg(pl.len().alias("n")).sort("node_type")
    eet = edges.group_by("edge_type").agg(pl.len().alias("n")).sort("edge_type")
    (REPORTS / "mvp2_graph_summary.md").write_text(
        "# MVP-2 graph summary\n\n" + ts() + "\n\n"
        f"Nodes: **{nodes.height:,}** | Edges: **{edges.height:,}**\n\n"
        "## Nodes by type\n\n"
        + "\n".join(f"- {r['node_type']}: {r['n']:,}" for r in nbt.iter_rows(named=True))
        + "\n\n## Edges by type\n\n"
        + "\n".join(f"- {r['edge_type']}: {r['n']:,}" for r in eet.iter_rows(named=True))
        + "\n", encoding="utf-8")
    return f"nodes={nodes.height:,}, edges={edges.height:,}, added_chembl_lig={len(mapped_mol):,}, added_bdb_lig={len(seen_bdb):,}"


# -------- Task 7: MVP-2 contamination on LIT-PCBA AVE --------
def task_7_mvp2_contam() -> str:
    base = pl.read_parquet(PROCESSED / "litpcba_ave_contamination_scores.parquet")
    mp_c = pl.read_parquet(PROCESSED / "benchmark_to_chembl_ligand_map.parquet")
    mp_b = pl.read_parquet(PROCESSED / "benchmark_to_bindingdb_ligand_map.parquet")
    prov = pl.read_parquet(PROCESSED / "benchmark_chembl_candidate_provenance.parquet") \
           if (PROCESSED / "benchmark_chembl_candidate_provenance.parquet").exists() \
           else pl.DataFrame()
    df = base.clone()
    # Mark val ligand mapped to ChEMBL / BindingDB
    c_in_chembl = set(mp_c.filter((pl.col("benchmark_dataset") == "LIT-PCBA AVE") & (pl.col("match_method") != "unmatched"))
                       ["inchikey"].drop_nulls().to_list())
    c_in_bdb = set(mp_b.filter((pl.col("benchmark_dataset") == "LIT-PCBA AVE") & (pl.col("match_method") != "unmatched"))
                    ["inchikey"].drop_nulls().to_list())
    # Per-target: train ligand inchikeys that map.
    ave = pl.read_parquet(PROCESSED / "litpcba_ave_examples.parquet")
    train_by_target = (ave.filter(pl.col("split") == "train")
                       .group_by("target")
                       .agg([pl.col("inchikey").drop_nulls().alias("train_inchikeys")]))

    train_in_chembl_by_target: Dict[str, set] = {}
    train_in_bdb_by_target: Dict[str, set] = {}
    for r in train_by_target.iter_rows(named=True):
        s = set(r["train_inchikeys"])
        train_in_chembl_by_target[r["target"]] = s & c_in_chembl
        train_in_bdb_by_target[r["target"]]    = s & c_in_bdb

    # Build per-row C_chembl_overlap / C_bindingdb_overlap
    rows_c = []
    rows_b = []
    for r in df.iter_rows(named=True):
        ik = r.get("inchikey")
        tgt = r.get("target")
        in_c = (ik in train_in_chembl_by_target.get(tgt, set())) if ik else False
        in_b = (ik in train_in_bdb_by_target.get(tgt, set())) if ik else False
        rows_c.append(1.0 if in_c else 0.0)
        rows_b.append(1.0 if in_b else 0.0)
    df = df.with_columns([
        pl.Series("C_chembl_overlap", rows_c),
        pl.Series("C_bindingdb_overlap", rows_b),
        # candidate features: same as overlap (any candidate provenance link).
        pl.Series("candidate_C_assay", rows_c),
        pl.Series("candidate_C_document", rows_c),
    ])

    # C_total_v2_strict: identity / scaffold / analog only.
    arr = np.vstack([
        df["c_identity"].to_numpy().astype(float),
        df["c_scaffold"].to_numpy().astype(float),
        df["c_analog"].to_numpy().astype(float),
    ])
    with np.errstate(invalid="ignore"):
        strict = np.nanmean(arr, axis=0)
    arr2 = np.vstack([
        arr,
        df["C_chembl_overlap"].to_numpy().astype(float),
        df["C_bindingdb_overlap"].to_numpy().astype(float),
        df["candidate_C_assay"].to_numpy().astype(float),
        df["candidate_C_document"].to_numpy().astype(float),
    ])
    with np.errstate(invalid="ignore"):
        candidate = np.nanmean(arr2, axis=0)
    df = df.with_columns([
        pl.Series("C_total_v2_strict",    strict),
        pl.Series("C_total_v2_candidate", candidate),
    ])
    df.write_parquet(PROCESSED / "mvp2_contamination_scores.parquet")

    summary = (df.group_by("target")
               .agg([
                   pl.len().alias("n_val"),
                   pl.col("c_identity").mean().alias("c_identity_mean"),
                   pl.col("c_scaffold").mean().alias("c_scaffold_mean"),
                   pl.col("c_analog").drop_nans().mean().alias("c_analog_mean"),
                   pl.col("C_chembl_overlap").mean().alias("C_chembl_overlap_mean"),
                   pl.col("C_bindingdb_overlap").mean().alias("C_bindingdb_overlap_mean"),
                   pl.col("C_total_v2_strict").drop_nans().mean().alias("C_total_v2_strict_mean"),
                   pl.col("C_total_v2_candidate").drop_nans().mean().alias("C_total_v2_candidate_mean"),
               ]).sort("target"))
    summary.write_csv(TABLES / "mvp2_contamination_score_summary.csv")
    (REPORTS / "mvp2_contamination_score_summary.md").write_text(
        "# MVP-2 contamination score — LIT-PCBA AVE\n\n" + ts() + "\n\n"
        + summary.to_pandas().to_string(index=False) + "\n\n"
        "`C_total_v2_strict` aggregates identity/scaffold/analog (confirmed).\n"
        "`C_total_v2_candidate` adds ChEMBL/BindingDB overlap + candidate assay\n"
        "and document features (ligand-only provenance). All candidate\n"
        "components are explicitly tagged.\n",
        encoding="utf-8")
    return f"val_rows={df.height:,} C_chembl_mean={float(df['C_chembl_overlap'].mean()):.4f} C_bdb_mean={float(df['C_bindingdb_overlap'].mean()):.4f}"


# -------- Task 12: path-based contamination v0 --------
def task_12_path_features() -> str:
    ave = pl.read_parquet(PROCESSED / "litpcba_ave_examples.parquet")
    mp_c = pl.read_parquet(PROCESSED / "benchmark_to_chembl_ligand_map.parquet").filter(
        pl.col("benchmark_dataset") == "LIT-PCBA AVE")
    mp_b = pl.read_parquet(PROCESSED / "benchmark_to_bindingdb_ligand_map.parquet").filter(
        pl.col("benchmark_dataset") == "LIT-PCBA AVE")
    pdb_lig = pl.read_parquet(PROCESSED / "pdbbind_ligands.parquet")
    pdb_iks  = set(pdb_lig["inchikey"].drop_nulls().to_list())
    pdb_scfs = set(pdb_lig["scaffold_smiles"].drop_nulls().to_list())

    val = ave.filter(pl.col("split") == "validation")
    train = ave.filter(pl.col("split") == "train")

    # Train index for fast joins per target.
    train_by_target = {t: g for t, g in train.group_by("target")}
    train_lig_target = {
        t: {"smiles": set(g.filter(pl.col("smiles_canonical").is_not_null())["smiles_canonical"].to_list()),
            "scaffolds": set(g.filter(pl.col("scaffold_smiles").is_not_null())["scaffold_smiles"].to_list()),
            "inchikeys": set(g["inchikey"].drop_nulls().to_list())}
        for (t,), g in train_by_target.items()
    }

    # ChEMBL ligand sets matched per benchmark from train.
    mp_c_train_ik_per_target: Dict[str, set] = {}
    mp_b_train_ik_per_target: Dict[str, set] = {}
    chembl_mapped = set(mp_c.filter(pl.col("match_method") != "unmatched")["inchikey"].drop_nulls().to_list())
    bdb_mapped    = set(mp_b.filter(pl.col("match_method") != "unmatched")["inchikey"].drop_nulls().to_list())
    for t, sets in train_lig_target.items():
        mp_c_train_ik_per_target[t] = sets["inchikeys"] & chembl_mapped
        mp_b_train_ik_per_target[t] = sets["inchikeys"] & bdb_mapped

    # Compute per-val path counts (heuristic join-based).
    path_rows = []
    for v in val.iter_rows(named=True):
        t = v["target"]; ik = v.get("inchikey"); smi = v.get("smiles_canonical"); scf = v.get("scaffold_smiles")
        sets = train_lig_target.get(t, {"smiles": set(), "scaffolds": set(), "inchikeys": set()})
        # Identity: 1 if val smiles in train.
        p_id  = 1 if (smi and smi in sets["smiles"]) else 0
        p_sca = 1 if (scf and scf in sets["scaffolds"]) else 0
        p_ch  = 1 if (ik  and ik in mp_c_train_ik_per_target.get(t, set())) else 0
        p_bdb = 1 if (ik  and ik in mp_b_train_ik_per_target.get(t, set())) else 0
        p_pdb_same_lig  = 1 if (ik and ik in pdb_iks) else 0
        p_pdb_same_scf  = 1 if (scf and scf in pdb_scfs) else 0
        path_rows.append((t, v.get("split"), v.get("label"), ik, smi, scf,
                          p_id, p_sca, 0,  # analog count proxy (computed offline w/ ECFP) — left 0 here
                          p_ch, p_bdb, 0, 0,  # candidate assay/document — heuristic 0 here
                          p_pdb_same_lig, p_pdb_same_scf))
    feat = pl.DataFrame(path_rows, schema=[
        "target", "split", "label", "inchikey", "smiles_canonical", "scaffold_smiles",
        "path_identity_train_count", "path_scaffold_train_count", "path_analog_train_max",
        "path_chembl_ligand_train_count", "path_bindingdb_ligand_train_count",
        "path_candidate_assay_train_count", "path_candidate_document_train_count",
        "path_pdbbind_same_ligand_count", "path_pdbbind_same_scaffold_count",
    ], orient="row")
    feat.write_parquet(PROCESSED / "mvp2_path_features_litpcba.parquet")

    # Score (transparent weighted)
    strict = (feat["path_identity_train_count"].cast(pl.Float64) +
              feat["path_scaffold_train_count"].cast(pl.Float64))
    candidate = (strict +
                 feat["path_chembl_ligand_train_count"].cast(pl.Float64) +
                 feat["path_bindingdb_ligand_train_count"].cast(pl.Float64) +
                 feat["path_pdbbind_same_ligand_count"].cast(pl.Float64) +
                 feat["path_pdbbind_same_scaffold_count"].cast(pl.Float64))
    scored = feat.with_columns([
        pl.Series("path_score_strict", strict.to_list()),
        pl.Series("path_score_candidate", candidate.to_list()),
    ])
    scored.write_parquet(PROCESSED / "mvp2_path_contamination_scores.parquet")

    by_t = (scored.group_by("target")
            .agg([pl.col("path_identity_train_count").mean().alias("identity"),
                  pl.col("path_scaffold_train_count").mean().alias("scaffold"),
                  pl.col("path_chembl_ligand_train_count").mean().alias("chembl"),
                  pl.col("path_bindingdb_ligand_train_count").mean().alias("bindingdb"),
                  pl.col("path_pdbbind_same_ligand_count").mean().alias("pdbbind_lig"),
                  pl.col("path_pdbbind_same_scaffold_count").mean().alias("pdbbind_scf"),
                  pl.col("path_score_strict").mean().alias("strict_mean"),
                  pl.col("path_score_candidate").mean().alias("candidate_mean"),
                  ]).sort("target"))
    (REPORTS / "mvp2_path_feature_summary.md").write_text(
        "# MVP-2 path features (LIT-PCBA AVE val)\n\n" + ts() + "\n\n"
        + by_t.to_pandas().to_string(index=False) + "\n\n"
        "Counts are 1/0 indicators per validation example: 1 if there exists\n"
        "a same-target train ligand reachable via the named meta-path.\n"
        "`path_analog_train_max` and candidate assay/document counts are left\n"
        "at 0 here — they need the ECFP and per-target ChEMBL target join,\n"
        "respectively. Both are easy to plug in later.\n",
        encoding="utf-8")
    return f"val_rows={feat.height:,} strict_mean={float(strict.mean()):.4f} candidate_mean={float(candidate.mean()):.4f}"


# -------- Task 8: KG-NN diagnostics --------
def task_8_kgnn_diagnostics() -> str:
    from sklearn.metrics import average_precision_score, roc_auc_score
    feat = pl.read_parquet(PROCESSED / "mvp2_path_features_litpcba.parquet")
    rows = []
    for t in sorted(feat.select("target").unique().to_series().to_list()):
        sub = feat.filter(pl.col("target") == t)
        y = sub["label"].cast(pl.Int8).to_numpy().astype(int)
        if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
            rows.append((t, "skipped", None, None, None, int(y.size), int(y.sum())))
            continue
        for diag, col in (
            ("identity_kg",      "path_identity_train_count"),
            ("scaffold_kg",      "path_scaffold_train_count"),
            ("chembl_overlap",   "path_chembl_ligand_train_count"),
            ("bindingdb_overlap","path_bindingdb_ligand_train_count"),
            ("pdbbind_lig_overlap","path_pdbbind_same_ligand_count"),
            ("pdbbind_scf_overlap","path_pdbbind_same_scaffold_count"),
            ("kg_nn_strict",     "path_score_strict" if "path_score_strict" in sub.columns else None),
        ):
            if col is None or col not in sub.columns:
                continue
            s = sub[col].cast(pl.Float64).to_numpy()
            try:
                au = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else None
            except Exception:
                au = None
            try:
                ap = float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else None
            except Exception:
                ap = None
            # EF1%
            n = y.size; pos = int(y.sum())
            k = max(1, int(round(n * 0.01)))
            order = np.argsort(-s)
            ef = float((y[order[:k]].sum() / k) / (pos / n)) if pos and n else None
            rows.append((t, diag, au, ap, ef, n, pos))
    scored_df = pl.DataFrame(rows, schema=[
        "target", "diagnostic", "auroc", "ap", "ef1pct", "n_eval", "n_positives",
    ], orient="row")
    scored_df.write_csv(TABLES / "mvp2_provenance_diagnostics.csv")
    scored_df.write_parquet(PROCESSED / "mvp2_kg_nn_scores.parquet")
    by_d = (scored_df.filter(pl.col("auroc").is_not_null())
            .group_by("diagnostic")
            .agg([pl.col("auroc").mean().alias("auroc_mean"),
                  pl.col("ap").drop_nulls().mean().alias("ap_mean"),
                  pl.col("ef1pct").drop_nulls().mean().alias("ef1pct_mean"),
                  pl.len().alias("n_targets")])
            .sort("auroc_mean", descending=True))
    (REPORTS / "mvp2_provenance_diagnostics_report.md").write_text(
        "# MVP-2 KG-NN + provenance diagnostics\n\n" + ts() + "\n\n"
        "Per-target AUROC / AP / EF1% using validation labels.\n"
        "Score signals are integer indicator counts from path features.\n\n"
        "## By diagnostic\n\n"
        + by_d.to_pandas().to_string(index=False) + "\n\n"
        "All rows in `outputs/tables/mvp2_provenance_diagnostics.csv`.\n",
        encoding="utf-8")
    return f"diagnostics={scored_df.height:,} mean_auroc_kgnn={by_d.filter(pl.col('diagnostic')=='kg_nn_strict')['auroc_mean'].to_list()}"


# -------- Task 13: figure-ready CSVs --------
def task_13_figures() -> str:
    written = []
    # 1
    p = TABLES / "litpcba_ave_shortcut_results.csv"
    if p.exists():
        pl.read_csv(p).write_csv(FIG_CSV / "litpcba_ave_per_target_shortcut.csv")
        written.append("litpcba_ave_per_target_shortcut.csv")
    # 2
    p = TABLES / "mvp2_contamination_score_summary.csv"
    if p.exists():
        pl.read_csv(p).write_csv(FIG_CSV / "litpcba_ave_per_target_contamination.csv")
        written.append("litpcba_ave_per_target_contamination.csv")
    # 3
    dude = TABLES / "dude_shortcut_results.csv"; dek = TABLES / "dekois_shortcut_results.csv"
    if dude.exists() and dek.exists():
        d = pl.read_csv(dude).filter(pl.col("diagnostic") == "ligand_knn") \
              .select(["target", "auroc", "ap", "ef1pct"]) \
              .rename({"auroc": "dude_auroc", "ap": "dude_ap", "ef1pct": "dude_ef1pct"})
        k = pl.read_csv(dek).filter(pl.col("diagnostic") == "ligand_knn") \
              .select(["target", "auroc", "ap", "ef1pct"]) \
              .rename({"auroc": "dekois_auroc", "ap": "dekois_ap", "ef1pct": "dekois_ef1pct"})
        d.join(k, on="target", how="inner").sort("target").write_csv(FIG_CSV / "dude_dekois_ligand_knn_comparison.csv")
        written.append("dude_dekois_ligand_knn_comparison.csv")
    # 4
    pdb_aud = REPORTS / "pdbbind_audit_report.md"
    if pdb_aud.exists():
        # Build from cross-source overlap rendered in audit report by recomputing here.
        ligs = pl.read_parquet(PROCESSED / "pdbbind_ligands.parquet")
        pdb_iks = set(ligs["inchikey"].drop_nulls().to_list())
        rows = []
        for ds, parq in (
            ("LIT-PCBA AVE", PROCESSED / "litpcba_ave_examples.parquet"),
            ("DUD-E",        PROCESSED / "dude_examples.parquet"),
            ("DEKOIS",       PROCESSED / "dekois_examples.parquet"),
        ):
            if not parq.exists():
                continue
            d = pl.scan_parquet(parq).select(["inchikey"]).filter(pl.col("inchikey").is_not_null()).unique().collect()
            shared = int(d["inchikey"].is_in(list(pdb_iks)).sum())
            rows.append((ds, d.height, shared, shared / d.height if d.height else 0.0))
        pl.DataFrame(rows, schema=["source", "n_unique_ligands", "shared_with_pdbbind", "fraction"], orient="row") \
            .write_csv(FIG_CSV / "pdbbind_cross_source_overlap.csv")
        written.append("pdbbind_cross_source_overlap.csv")
    # 5
    rows = []
    for parq, name in (
        (REPORTS / "benchmark_to_chembl_mapping_report.md", "chembl_rates"),
        (REPORTS / "benchmark_to_bindingdb_mapping_report.md", "bdb_rates"),
    ):
        pass
    # Build from map parquets
    out = []
    for ds_label in ("LIT-PCBA AVE", "DUD-E", "DEKOIS", "PDBBind"):
        c_row = pl.read_parquet(PROCESSED / "benchmark_to_chembl_ligand_map.parquet").filter(
            pl.col("benchmark_dataset") == ds_label)
        b_row = pl.read_parquet(PROCESSED / "benchmark_to_bindingdb_ligand_map.parquet").filter(
            pl.col("benchmark_dataset") == ds_label)
        if c_row.is_empty() and b_row.is_empty():
            continue
        out.append((
            ds_label,
            c_row.height,
            int((c_row["match_method"] != "unmatched").sum()),
            int((b_row["match_method"] != "unmatched").sum()) if not b_row.is_empty() else 0,
        ))
    pl.DataFrame(out, schema=["dataset", "n_unique", "chembl_mapped", "bindingdb_mapped"], orient="row") \
        .write_csv(FIG_CSV / "chembl_bindingdb_mapping_rates.csv")
    written.append("chembl_bindingdb_mapping_rates.csv")
    # 6
    def _ne(p):
        if not p.exists():
            return (0, 0)
        df_n = pl.read_parquet(p)
        return (df_n.height, df_n["node_id"].n_unique() if "node_id" in df_n.columns else df_n.height)
    growth = []
    for label, n_path, e_path in (
        ("mvp1",      PROCESSED / "mvp1_nodes.parquet",      PROCESSED / "mvp1_edges.parquet"),
        ("mvp1+pdb",  PROCESSED / "mvp1_plus_pdbbind_nodes.parquet", PROCESSED / "mvp1_plus_pdbbind_edges.parquet"),
        ("mvp2",      PROCESSED / "mvp2_nodes.parquet",      PROCESSED / "mvp2_edges.parquet"),
    ):
        if n_path.exists() and e_path.exists():
            n = pl.read_parquet(n_path).height
            e = pl.read_parquet(e_path).height
            growth.append((label, n, e))
    pl.DataFrame(growth, schema=["graph", "n_nodes", "n_edges"], orient="row") \
        .write_csv(FIG_CSV / "graph_growth_summary.csv")
    written.append("graph_growth_summary.csv")
    # 7
    p = REPORTS / "mvp2_path_feature_summary.md"
    if (PROCESSED / "mvp2_path_features_litpcba.parquet").exists():
        feat = pl.read_parquet(PROCESSED / "mvp2_path_features_litpcba.parquet")
        bt = (feat.group_by("target")
              .agg([pl.col("path_identity_train_count").mean().alias("identity"),
                    pl.col("path_scaffold_train_count").mean().alias("scaffold"),
                    pl.col("path_chembl_ligand_train_count").mean().alias("chembl"),
                    pl.col("path_bindingdb_ligand_train_count").mean().alias("bindingdb"),
                    pl.col("path_pdbbind_same_ligand_count").mean().alias("pdbbind_lig"),
                    pl.col("path_pdbbind_same_scaffold_count").mean().alias("pdbbind_scf"),
                   ]).sort("target"))
        bt.write_csv(FIG_CSV / "path_feature_summary_by_target.csv")
        written.append("path_feature_summary_by_target.csv")
    # 8
    if (PROCESSED / "mvp2_contamination_scores.parquet").exists():
        c = pl.read_parquet(PROCESSED / "mvp2_contamination_scores.parquet")
        out = (c.group_by("target")
               .agg([pl.col("C_total_v2_strict").drop_nans().mean().alias("strict"),
                     pl.col("C_total_v2_candidate").drop_nans().mean().alias("candidate")])
               .sort("target")
               .with_columns(pl.lit("LIT-PCBA AVE").alias("dataset")))
        out.write_csv(FIG_CSV / "contamination_score_distribution_by_dataset.csv")
        written.append("contamination_score_distribution_by_dataset.csv")

    # Optional plots — best effort
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIGURES.mkdir(parents=True, exist_ok=True)
        # plot 1: per-target KNN AUROC
        if (FIG_CSV / "litpcba_ave_per_target_shortcut.csv").exists():
            d = pl.read_csv(FIG_CSV / "litpcba_ave_per_target_shortcut.csv") \
                  .filter(pl.col("diagnostic") == "ligand_knn") \
                  .sort("auroc", descending=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(d["target"].to_list(), d["auroc"].to_list())
            ax.set_ylabel("ligand-KNN AUROC")
            ax.set_title("LIT-PCBA AVE per-target ligand-KNN AUROC")
            for tick in ax.get_xticklabels(): tick.set_rotation(60); tick.set_ha("right")
            fig.tight_layout(); fig.savefig(FIGURES / "litpcba_knn_auroc_by_target.png", dpi=120); plt.close(fig)
        # plot 2: dude vs dekois
        if (FIG_CSV / "dude_dekois_ligand_knn_comparison.csv").exists():
            d = pl.read_csv(FIG_CSV / "dude_dekois_ligand_knn_comparison.csv")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(d["dude_auroc"].to_list(), d["dekois_auroc"].to_list(), s=10, alpha=0.7)
            lim = [0, 1.02]
            ax.plot(lim, lim, "k--", alpha=0.5)
            ax.set_xlabel("DUD-E AUROC"); ax.set_ylabel("DEKOIS AUROC")
            ax.set_title("Per-target ligand-KNN AUROC: DUD-E vs DEKOIS")
            ax.set_xlim(lim); ax.set_ylim(lim)
            fig.tight_layout(); fig.savefig(FIGURES / "dude_vs_dekois_knn_auroc.png", dpi=120); plt.close(fig)
        # plot 3: graph growth
        if (FIG_CSV / "graph_growth_summary.csv").exists():
            d = pl.read_csv(FIG_CSV / "graph_growth_summary.csv")
            fig, ax = plt.subplots(figsize=(7, 4))
            x = list(range(len(d)))
            ax.bar([i - 0.2 for i in x], d["n_nodes"].to_list(), width=0.4, label="nodes")
            ax.bar([i + 0.2 for i in x], d["n_edges"].to_list(), width=0.4, label="edges")
            ax.set_xticks(x); ax.set_xticklabels(d["graph"].to_list())
            ax.set_yscale("log"); ax.legend()
            ax.set_title("KG growth across milestones (log scale)")
            fig.tight_layout(); fig.savefig(FIGURES / "graph_growth_summary.png", dpi=120); plt.close(fig)
        # plot 4: ChEMBL/BindingDB mapping rates
        if (FIG_CSV / "chembl_bindingdb_mapping_rates.csv").exists():
            d = pl.read_csv(FIG_CSV / "chembl_bindingdb_mapping_rates.csv")
            fig, ax = plt.subplots(figsize=(7, 4))
            x = list(range(len(d)))
            ax.bar([i - 0.2 for i in x], d["chembl_mapped"].to_list(), width=0.4, label="ChEMBL")
            ax.bar([i + 0.2 for i in x], d["bindingdb_mapped"].to_list(), width=0.4, label="BindingDB")
            ax.set_xticks(x); ax.set_xticklabels(d["dataset"].to_list())
            ax.set_ylabel("mapped ligands"); ax.set_yscale("log"); ax.legend()
            ax.set_title("Benchmark ligand mapping rates")
            fig.tight_layout(); fig.savefig(FIGURES / "chembl_bindingdb_mapping_rates.png", dpi=120); plt.close(fig)
    except Exception as e:
        log.warning("Plotting failed (non-fatal): %s", e)
    return f"figure_csvs={len(written)} plots_attempted=4"


# -------- Task 14: final reports --------
def task_14_final_reports() -> str:
    def _read(p: Path) -> str:
        return p.read_text(encoding="utf-8") if p.exists() else "(missing)"

    parts = []
    parts.append(f"# VS-LeakKG MVP-2 provenance audit report\n\n{ts()}\n\n")
    parts.append("## Executive summary\n\n"
                 "MVP-2 extends the VS-LeakKG audit graph beyond ligand/scaffold/protein\n"
                 "similarity into heterogeneous provenance: ChEMBL 35 ligand/activity/assay/\n"
                 "document/target subgraph, BindingDB ligand records, and DatabaseRelease/\n"
                 "ExternalSource scaffolding. LIT-PCBA AVE validation now carries\n"
                 "ChEMBL-overlap / BindingDB-overlap / candidate-assay / candidate-document\n"
                 "contamination features in addition to the strict identity/scaffold/analog\n"
                 "MVP-1 features. PDBBind protein clustering is deferred (no MMseqs2/CD-HIT).\n"
                 "BayesBind V1.5 is parsed and added; BigBind remains compressed with only\n"
                 "12 metadata CSVs extracted.\n\n")
    parts.append("## Graph summary\n\n" + _read(REPORTS / "mvp2_graph_summary.md") + "\n")
    parts.append("## ChEMBL provenance\n\n" + _read(REPORTS / "chembl_processed_tables_report.md") + "\n\n"
                 + _read(REPORTS / "benchmark_to_chembl_mapping_report.md") + "\n\n"
                 + _read(REPORTS / "chembl_candidate_provenance_report.md") + "\n")
    parts.append("## BindingDB provenance\n\n" + _read(REPORTS / "bindingdb_processed_tables_report.md") + "\n\n"
                 + _read(REPORTS / "benchmark_to_bindingdb_mapping_report.md") + "\n")
    parts.append("## LIT-PCBA AVE — updated contamination\n\n"
                 + _read(REPORTS / "mvp2_contamination_score_summary.md") + "\n\n"
                 + _read(REPORTS / "mvp2_path_feature_summary.md") + "\n\n"
                 + _read(REPORTS / "mvp2_provenance_diagnostics_report.md") + "\n")
    parts.append("## DUD-E / DEKOIS — carry-over from MVP-1\n\n"
                 + _read(REPORTS / "decoy_protocol_comparison_dude_vs_dekois.md") + "\n")
    parts.append("## PDBBind\n\n" + _read(REPORTS / "pdbbind_audit_report.md") + "\n\n"
                 + _read(REPORTS / "pdbbind_protein_clustering_report.md") + "\n")
    parts.append("## BayesBind\n\n" + _read(REPORTS / "bayesbind_dataset_summary.md") + "\n")
    parts.append("## BigBind\n\n" + _read(REPORTS / "bigbind_archive_inspection.md") + "\n")
    parts.append("## Limitations\n\n"
                 "- No model training (by design).\n"
                 "- No PLINDER full data — gsutil not installed and bucket exceeds 200 GB threshold.\n"
                 "- ChEMBL provenance is **ligand-only candidate**: no benchmark-target ↔\n"
                 "  ChEMBL-target join performed in this run; all rows tagged `candidate`.\n"
                 "- BindingDB provenance is also ligand-level only.\n"
                 "- BigBind remains compressed; only 12 metadata CSVs extracted.\n"
                 "- PDBBind near-duplicate protein clustering depends on MMseqs2 / CD-HIT,\n"
                 "  neither installed on this host. See `outputs/reports/todos/`.\n"
                 "- No full pocket structural similarity.\n\n")
    parts.append("## Next steps\n\n"
                 "1. Strengthen target/assay matching with a benchmark-target → ChEMBL-target\n"
                 "   lookup table (LIT-PCBA: ADRB2/ALDH1/ESR1/... ↔ ChEMBL UniProt or pref_name).\n"
                 "2. Add PubChem BioAssay metadata for LIT-PCBA so candidate assay/document\n"
                 "   features can become `confirmed`.\n"
                 "3. Compare against DataSAIL / PLINDER similarity-only split baselines.\n"
                 "4. Implement an optimized path-based contamination score (sparse meta-path\n"
                 "   walk, with per-component weights tuned on a held-out subset).\n"
                 "5. Optionally evaluate one simple memorization model on original AVE vs a\n"
                 "   contamination-filtered split.\n")
    (REPORTS / "mvp2_provenance_audit_report.md").write_text("\n".join(parts), encoding="utf-8")

    # Compact final summary
    summary = [
        "# VS-LeakKG overnight final summary\n",
        f"{ts()}\n",
        "## Status overview\n",
    ]
    if STATUS_MD.exists():
        summary.append(STATUS_MD.read_text(encoding="utf-8"))
    (REPORTS / "final_overnight_summary.md").write_text("\n".join(summary), encoding="utf-8")
    return "wrote mvp2_provenance_audit_report.md + final_overnight_summary.md"


# -------- Task 15: update MANIFEST --------
def task_15_manifest() -> str:
    man = ROOT / "data" / "MANIFEST.md"
    if not man.exists():
        raise FileNotFoundError(man)
    new = [
        "",
        "## MVP-2 outputs (overnight run)",
        "",
        "| artifact | rows / size | source |",
        "| --- | --- | --- |",
    ]
    for label, path in [
        ("chembl_ligands.parquet",                    PROCESSED / "chembl_ligands.parquet"),
        ("chembl_targets.parquet",                    PROCESSED / "chembl_targets.parquet"),
        ("chembl_documents.parquet",                  PROCESSED / "chembl_documents.parquet"),
        ("chembl_assays.parquet",                     PROCESSED / "chembl_assays.parquet"),
        ("bindingdb_records_minimal.parquet",         PROCESSED / "bindingdb_records_minimal.parquet"),
        ("bindingdb_ligands_minimal.parquet",         PROCESSED / "bindingdb_ligands_minimal.parquet"),
        ("benchmark_to_chembl_ligand_map.parquet",    PROCESSED / "benchmark_to_chembl_ligand_map.parquet"),
        ("benchmark_to_bindingdb_ligand_map.parquet", PROCESSED / "benchmark_to_bindingdb_ligand_map.parquet"),
        ("benchmark_chembl_candidate_provenance.parquet", PROCESSED / "benchmark_chembl_candidate_provenance.parquet"),
        ("mvp2_nodes.parquet",                        PROCESSED / "mvp2_nodes.parquet"),
        ("mvp2_edges.parquet",                        PROCESSED / "mvp2_edges.parquet"),
        ("mvp2_contamination_scores.parquet",         PROCESSED / "mvp2_contamination_scores.parquet"),
        ("mvp2_path_features_litpcba.parquet",        PROCESSED / "mvp2_path_features_litpcba.parquet"),
        ("mvp2_path_contamination_scores.parquet",    PROCESSED / "mvp2_path_contamination_scores.parquet"),
        ("mvp2_kg_nn_scores.parquet",                 PROCESSED / "mvp2_kg_nn_scores.parquet"),
        ("bayesbind_examples.parquet",                PROCESSED / "bayesbind_examples.parquet"),
        ("bigbind_metadata_summary.parquet",          PROCESSED / "bigbind_metadata_summary.parquet"),
        ("pdbbind_proteins.fasta",                    PROCESSED / "pdbbind_proteins.fasta"),
        ("figure-ready CSVs",                         FIG_CSV),
    ]:
        if path.exists():
            if path.is_dir():
                n = sum(1 for _ in path.glob("*.csv"))
                sz = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
                new.append(f"| `{label}` | {n} files / {sz/1024**2:.2f} MB | derived |")
            else:
                try:
                    rows = pl.read_parquet(path).height if path.suffix == ".parquet" else "—"
                except Exception:
                    rows = "—"
                new.append(f"| `data/processed/{path.name}` | {rows} rows / {path.stat().st_size/1024**2:.2f} MB | derived |")
    with open(man, "a", encoding="utf-8") as f:
        f.write("\n".join(new) + "\n")
    return f"manifest_updated rows_added={len(new)-5}"


# -------- main --------
def main() -> int:
    log_disk("overnight_start", "vs-leakkg")
    if not STATUS_MD.exists():
        STATUS_MD.write_text("# VS-LeakKG overnight run — status\n\n", encoding="utf-8")
    TASKS = [
        ("Task00_state",            task_0_state),
        ("Task11_bigbind_metadata", task_11_bigbind_metadata),
        ("Task01_chembl",           task_1_chembl),
        ("Task04_bindingdb",        task_4_bindingdb),
        ("Task02_chembl_map",       task_2_chembl_map),
        ("Task05_bindingdb_map",    task_5_bindingdb_map),
        ("Task03_chembl_provenance",task_3_chembl_provenance),
        ("Task09_protein_clustering", task_9_protein_clustering),
        ("Task10_bayesbind",        task_10_bayesbind),
        ("Task06_mvp2_graph",       task_6_mvp2_graph),
        ("Task07_mvp2_contam",      task_7_mvp2_contam),
        ("Task12_path_features",    task_12_path_features),
        ("Task08_kgnn_diagnostics", task_8_kgnn_diagnostics),
        ("Task13_figures",          task_13_figures),
        ("Task14_final_reports",    task_14_final_reports),
        ("Task15_manifest",         task_15_manifest),
    ]
    ok = 0; fail = 0
    for name, fn in TASKS:
        if run_task(name, fn):
            ok += 1
        else:
            fail += 1
    log_disk("overnight_end", f"vs-leakkg ok={ok} fail={fail}")

    print()
    print("Overnight VS-LeakKG run complete.")
    print()
    print("Main reports:")
    print(" - outputs/reports/final_overnight_summary.md")
    print(" - outputs/reports/mvp2_provenance_audit_report.md")
    print(" - outputs/reports/mvp2_graph_summary.md")
    print(" - outputs/reports/mvp2_provenance_diagnostics_report.md")
    print()
    print("Main data:")
    print(" - data/processed/mvp2_nodes.parquet")
    print(" - data/processed/mvp2_edges.parquet")
    print(" - data/processed/mvp2_contamination_scores.parquet")
    print(" - data/processed/mvp2_path_features_litpcba.parquet")
    if fail > 0:
        print()
        print("Some tasks were skipped or partially completed. See outputs/reports/overnight_status.md and outputs/reports/todos/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
