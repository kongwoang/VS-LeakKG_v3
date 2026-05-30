"""Rebuild protein clusters across all KG protein sources.

Steps:
  1. Extract UniProt (accession, sequence) tuples from ChEMBL SQLite —
     gives ~16K proteins with sequences.
  2. Collect UniProt accessions used by BigBind / BindingDB / BayesBind
     (each row's `uniprot` / `uniprot_swissprot_id` columns). Map them
     back to sequences via ChEMBL where possible. Accessions not in ChEMBL
     are recorded as "missing" — they remain Protein nodes in the KG but
     won't participate in clustering edges. (Optional: fetch from UniProt
     REST API in a follow-up.)
  3. Write a deduplicated FASTA `data/processed/all_proteins.fasta`.
  4. Run `mmseqs easy-cluster` at 30, 50, 90 % min-seq-id.
  5. Convert each `*_cluster.tsv` to
     `data/processed/protein_clusters_{30,50,90}.parquet` with columns
     (accession, cluster_id).

The output parquet schema matches what `vsleakkg.kg.consolidate.
_add_protein_cluster_edges` expects: a per-row (member, cluster_representative)
pair, columns named flexibly.

Run:
    PYTHONPATH=src python -m vsleakkg.build_protein_clusters
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl

from vsleakkg import load_chembl_db

log = logging.getLogger("vsleakkg.build_protein_clusters")

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
CHEMBL_DB = RAW / "ChEMBL" / "extracted" / "chembl_35" / "chembl_35_sqlite" / "chembl_35.db"
REPORTS = ROOT / "outputs" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def collect_kg_uniprots() -> set[str]:
    """Return the set of UniProt accessions used by per-corpus parquets."""
    accs: set[str] = set()
    for fname, col in [
        ("bigbind_examples.parquet", "uniprot"),
        ("bayesbind_examples.parquet", "uniprot"),
        ("bindingdb_records_minimal.parquet", "uniprot_swissprot_id"),
    ]:
        p = PROCESSED / fname
        if not p.exists():
            log.warning("skip %s: parquet missing", fname)
            continue
        df = pl.read_parquet(p)
        if col in df.columns:
            vals = df.filter(pl.col(col).is_not_null())[col].unique().to_list()
            accs.update(v.strip() for v in vals if v and v.strip())
            log.info("  %s: %d unique UniProt accessions", fname, len(vals))
    return accs


def extract_chembl_sequences() -> pl.DataFrame:
    """Return a DataFrame with one row per (accession, sequence) from ChEMBL."""
    if not CHEMBL_DB.exists():
        raise FileNotFoundError(CHEMBL_DB)
    log.info("querying ChEMBL component_sequences (%s) ...", CHEMBL_DB)
    conn = load_chembl_db.connect(CHEMBL_DB)
    df = load_chembl_db.load_target_sequences(conn)
    conn.close()
    log.info("  raw rows: %d", df.height)
    # Drop duplicate accessions (same UniProt can be referenced by multiple tids).
    df = df.unique(subset=["accession"])
    df = df.filter(pl.col("accession").is_not_null() & pl.col("sequence").is_not_null())
    log.info("  unique (accession, sequence) pairs: %d", df.height)
    return df


def write_fasta(seqs: pl.DataFrame, out_path: Path) -> int:
    """Write deduplicated FASTA. Returns number of records written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for acc, seq in zip(seqs["accession"].to_list(),
                            seqs["sequence"].to_list()):
            if not acc or not seq:
                continue
            f.write(f">{acc}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")
            n += 1
    return n


def run_mmseqs(fasta: Path, out_prefix: Path, min_seq_id: float,
               coverage: float = 0.8, threads: int | None = None) -> Path:
    """Run `mmseqs easy-cluster` and return the *_cluster.tsv path."""
    threads = threads or (os.cpu_count() or 1)
    tmpdir = Path(tempfile.mkdtemp(prefix="mmseqs_tmp_"))
    try:
        cmd = ["mmseqs", "easy-cluster", str(fasta), str(out_prefix), str(tmpdir),
               "--min-seq-id", f"{min_seq_id:.2f}", "-c", str(coverage),
               "--cov-mode", "0",
               "--threads", str(threads)]
        log.info("running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    cluster_tsv = out_prefix.with_suffix(".tsv") if False else Path(str(out_prefix) + "_cluster.tsv")
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"mmseqs did not produce {cluster_tsv}")
    return cluster_tsv


def convert_cluster_tsv(tsv: Path, out_parquet: Path, resolution: str) -> int:
    """Convert MMseqs2 cluster.tsv -> parquet (member, cluster_id, resolution)."""
    # cluster.tsv: tab-separated, columns: representative \t member
    df = pl.read_csv(tsv, has_header=False, separator="\t",
                     new_columns=["cluster_id", "accession"])
    df = df.with_columns(pl.lit(resolution).alias("resolution"))
    df = df.select(["accession", "cluster_id", "resolution"])
    df.write_parquet(out_parquet)
    return df.height


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thresholds", default="30,50,90",
                    help="comma-separated %% identity thresholds")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    if shutil.which("mmseqs") is None:
        raise RuntimeError("mmseqs2 not found on PATH. Install via "
                           "`conda install -c bioconda mmseqs2`.")

    accs_in_kg = collect_kg_uniprots()
    log.info("KG uses %d unique UniProt accessions", len(accs_in_kg))

    chembl_seqs = extract_chembl_sequences()
    chembl_set = set(chembl_seqs["accession"].to_list())
    coverage = len(accs_in_kg & chembl_set)
    log.info("ChEMBL covers %d / %d (%.1f%%) of KG UniProts",
             coverage, len(accs_in_kg),
             100*coverage/len(accs_in_kg) if accs_in_kg else 0)

    # Union of all sequences we want to cluster: ChEMBL set (large pool) +
    # any KG accession we couldn't find (recorded but no cluster).
    fasta = PROCESSED / "all_proteins.fasta"
    n_written = write_fasta(chembl_seqs, fasta)
    log.info("wrote %s (%d sequences)", fasta, n_written)
    missing = accs_in_kg - chembl_set
    if missing:
        miss_log = REPORTS / "protein_clusters_missing_uniprots.txt"
        miss_log.write_text("\n".join(sorted(missing)) + "\n", encoding="utf-8")
        log.warning("%d KG UniProt accessions have no ChEMBL sequence (see %s)",
                    len(missing), miss_log)

    thresholds = [int(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    for thr in thresholds:
        min_seq_id = thr / 100.0
        out_prefix = PROCESSED / f"protein_clu_{thr}"
        cluster_tsv = run_mmseqs(fasta, out_prefix, min_seq_id, threads=args.threads)
        out_parquet = PROCESSED / f"protein_clusters_{thr}.parquet"
        n = convert_cluster_tsv(cluster_tsv, out_parquet, str(thr))
        n_clusters = pl.read_parquet(out_parquet)["cluster_id"].n_unique()
        log.info("threshold=%d: %d members across %d clusters -> %s",
                 thr, n, n_clusters, out_parquet)

    (REPORTS / "protein_cluster_build_report.md").write_text(
        f"# Protein cluster build\n\n"
        f"- KG UniProt accessions: {len(accs_in_kg)}\n"
        f"- ChEMBL sequences covered: {coverage} ({100*coverage/len(accs_in_kg):.1f}%)\n"
        f"- Missing accessions: {len(missing)} (see protein_clusters_missing_uniprots.txt)\n"
        f"- Thresholds: {args.thresholds}\n"
        f"- Outputs: protein_clusters_{{30,50,90}}.parquet\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
