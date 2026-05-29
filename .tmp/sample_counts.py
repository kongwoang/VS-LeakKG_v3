"""Tally sample counts per dataset from v3/data/processed/ parquets."""
from __future__ import annotations
import polars as pl
from pathlib import Path
import json

P = Path("/vol/dl-nguyenb5-solar/users/hoangpc/VS-LeakKG_v3/data/processed")

def safe(name: str):
    f = P / name
    if not f.exists():
        return None
    return pl.read_parquet(f)

rows = []

# --- Benchmarks with Example rows ---
for label, fname in [
    ("LIT-PCBA",      "litpcba_examples.parquet"),
    ("LIT-PCBA-AVE",  "litpcba_ave_examples.parquet"),
    ("DUD-E",         "dude_examples.parquet"),
    ("DEKOIS2",       "dekois_examples.parquet"),
    ("BayesBind",     "bayesbind_examples.parquet"),
]:
    df = safe(fname)
    if df is None:
        rows.append((label, "—", 0, 0, 0, "missing"))
        continue
    n = df.height
    cols = df.columns
    n_lig = df["smiles_canonical"].n_unique() if "smiles_canonical" in cols else (df["inchikey"].n_unique() if "inchikey" in cols else 0)
    tgt_col = "target" if "target" in cols else ("target_name" if "target_name" in cols else ("uniprot" if "uniprot" in cols else None))
    n_tgt = df[tgt_col].n_unique() if tgt_col else 0
    # label distribution
    lab_col = "label" if "label" in cols else ("label_value" if "label_value" in cols else None)
    if lab_col:
        try:
            actives = int((df[lab_col] == 1).sum())
            decoys = int((df[lab_col] == 0).sum())
            lab = f"act={actives:,} dec/inact={decoys:,}"
        except Exception:
            lab = f"col={lab_col}"
    else:
        lab = "n/a"
    rows.append((label, fname.split("_")[0], n, n_lig, n_tgt, lab))

# --- PDBBind (no Examples; row primitive = Complex) ---
df = safe("pdbbind_complexes.parquet")
if df is not None:
    n = df.height
    n_lig = df["ligand_id"].n_unique() if "ligand_id" in df.columns else 0
    n_tgt = df["uniprot"].n_unique() if "uniprot" in df.columns else (df["protein_id"].n_unique() if "protein_id" in df.columns else 0)
    rows.append(("PDBBind", "pdbbind", n, n_lig, n_tgt, "all act (binders)"))
else:
    rows.append(("PDBBind", "pdbbind", 0, 0, 0, "missing"))

# --- Reference DBs (counts, not "examples") ---
for label, fname, lig_col, tgt_col in [
    ("ChEMBL ligands",   "chembl_ligands.parquet",       "smiles_canonical", None),
    ("ChEMBL assays",    "chembl_assays.parquet",         None,                None),
    ("ChEMBL targets",   "chembl_targets.parquet",        None,                "uniprot"),
    ("ChEMBL documents", "chembl_documents.parquet",      None,                None),
    ("BindingDB ligands","bindingdb_ligands_minimal.parquet", "smiles_canonical", None),
    ("BindingDB records","bindingdb_records_minimal.parquet", None,            None),
]:
    df = safe(fname)
    if df is None:
        rows.append((label, fname, 0, 0, 0, "missing"))
        continue
    n = df.height
    n_lig = df[lig_col].n_unique() if lig_col and lig_col in df.columns else 0
    n_tgt = df[tgt_col].n_unique() if tgt_col and tgt_col in df.columns else 0
    rows.append((label, fname.split(".")[0], n, n_lig, n_tgt, ""))

# --- BigBind / PLINDER (no Python loader) ---
df = safe("bigbind_metadata_summary.parquet")
if df is not None:
    rows.append(("BigBind (metadata-only)", "bigbind_meta", df.height, 0, 0, "archive 18GB not extracted"))

print(json.dumps([{"name": r[0], "source": r[1], "n_rows": r[2], "n_unique_ligands": r[3], "n_unique_targets": r[4], "note": r[5]} for r in rows], indent=2))
