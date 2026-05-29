"""BayesBind V1.5 per-target loader.

Layout (per target):
  BayesBindV1.5/{test|val}/<TARGET_NAME>/
    actives.csv     # rich metadata: lig_smiles, standard_type, pchembl_value, uniprot, pocket, ...
    actives.smi     # SMILES, one per line
    random.csv      # decoys with pocket metadata
    random.smi
    pocket.pdb
    rec.pdb         # receptor variants
    rec_hs.pdb
    rec_nofix.pdb
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import polars as pl


def discover_targets(root: Path) -> List[tuple[str, str]]:
    """Returns list of (split, target_name). `split` is 'test' or 'val'."""
    base = root / "BayesBindV1.5"
    base = base if base.exists() else root
    out: List[tuple[str, str]] = []
    for split in ("test", "val"):
        sdir = base / split
        if not sdir.exists():
            continue
        for t in sorted(p for p in sdir.iterdir() if p.is_dir()):
            if (t / "actives.csv").exists() or (t / "random.csv").exists():
                out.append((split, t.name))
    return out


def load_target(root: Path, split: str, target: str) -> pl.DataFrame:
    base = root / "BayesBindV1.5"
    base = base if base.exists() else root
    tdir = base / split / target
    frames = []
    for fname, label, label_type in (
        ("actives.csv", 1, "active"),
        ("random.csv",  0, "random"),
    ):
        p = tdir / fname
        if not p.exists() or p.stat().st_size == 0:
            continue
        try:
            df = pl.read_csv(p, infer_schema_length=2000, ignore_errors=True)
        except Exception:
            continue
        if df.height == 0:
            continue
        # Standardize column name for SMILES (actives.csv uses lig_smiles; random.csv uses lig_smiles too).
        rename: dict = {}
        for c in df.columns:
            if c.lower() == "lig_smiles":
                rename[c] = "smiles_input"
        if rename:
            df = df.rename(rename)
        if "smiles_input" not in df.columns:
            continue
        df = df.with_columns([
            pl.lit("BayesBind").alias("source"),
            pl.lit(target).alias("target"),
            pl.lit(label, dtype=pl.Int8).alias("label"),
            pl.lit(label_type).alias("label_type"),
            pl.lit(split).alias("split"),
            pl.lit(fname).alias("source_file"),
        ])
        # Carry uniprot + pocket if available; otherwise null.
        keep_cols = ["smiles_input", "source", "target", "label", "label_type", "split", "source_file"]
        for opt in ("uniprot", "pocket", "standard_type", "standard_value",
                    "standard_units", "pchembl_value",
                    "ex_rec_pdb", "lig_cluster", "rec_cluster",
                    "num_pocket_residues"):
            if opt in df.columns:
                keep_cols.append(opt)
        frames.append(df.select(keep_cols))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def load_all(root: Path) -> pl.DataFrame:
    targets = discover_targets(root)
    parts = [load_target(root, split, t) for split, t in targets]
    parts = [p for p in parts if p.height > 0]
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed")
