"""Load PDBBind (v2020R1, re-processed in v2024) protein-ligand data.

The expected on-disk layout (the user's `data/raw/PBDBind/extracted/`):

    extracted/
      index/
        INDEX_general_PL.2020R1.lst    # 19,037 PL complexes
        INDEX_general_NL.2020R1.lst    # 143 NL  (out of scope here)
        INDEX_general_PN.2020R1.lst    # 1032 PN
        INDEX_general_PP.2020R1.lst    # 2798 PP
        README
      P-L/
        1981-2000/<pdb>/<pdb>_{ligand.mol2,ligand.sdf,pocket.pdb,protein.pdb}
        2001-2010/...
        2011-2019/...

Only the protein-ligand (PL) entries are loaded by this module.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import polars as pl


# Unit -> molar conversion factor.
_UNIT_TO_MOLAR = {
    "M":   1e0,
    "mM":  1e-3,
    "uM":  1e-6,
    "µM":  1e-6,
    "nM":  1e-9,
    "pM":  1e-12,
    "fM":  1e-15,
}

_AFFINITY_RE = re.compile(
    r"^\s*(?P<type>Kd|Ki|IC50)\s*(?P<cmp>=|<=|>=|<|>|~)\s*"
    r"(?P<val>[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*"
    r"(?P<unit>fM|pM|nM|uM|µM|mM|M)\s*$",
    re.IGNORECASE,
)

_LINE_RE = re.compile(
    r"^\s*(?P<pdb>\S+)\s+"
    r"(?P<res>\S+)\s+"
    r"(?P<year>\d{4})\s+"
    r"(?P<aff>\S[\S]*?)\s+//\s+"
    r"(?P<ref>\S+)(?:\s+\((?P<lig>[^)]+)\))?\s*"
    r"(?P<notes>.*)$"
)


def parse_affinity_string(raw: str) -> Dict[str, Optional[object]]:
    """Parse `Ki=0.068nM` -> structured dict including pK = -log10(value_M).
    Returns dict with affinity_type, comparator, value, unit, value_M, p_value,
    parse_ok."""
    if raw is None:
        return _aff_blank("none")
    s = raw.strip()
    m = _AFFINITY_RE.match(s)
    if not m:
        return _aff_blank(s)
    t = m.group("type")
    cmp_ = m.group("cmp")
    try:
        v = float(m.group("val"))
    except ValueError:
        return _aff_blank(s)
    unit = m.group("unit")
    unit_norm = "uM" if unit in ("µM",) else unit
    factor = _UNIT_TO_MOLAR.get(unit_norm)
    if factor is None or v <= 0:
        return _aff_blank(s)
    v_molar = v * factor
    p = -math.log10(v_molar)
    # Map type to canonical p-* name: Kd->pKd, Ki->pKi, IC50->pIC50
    pname = {"Kd": "pKd", "Ki": "pKi", "IC50": "pIC50"}[t]
    return {
        "affinity_raw": s,
        "affinity_type": t,
        "comparator": cmp_,
        "value": v,
        "unit": unit_norm,
        "value_M": v_molar,
        "p_value": p,
        "p_name": pname,
        "parse_ok": True,
    }


def _aff_blank(raw: str) -> Dict[str, Optional[object]]:
    return {
        "affinity_raw": raw,
        "affinity_type": None,
        "comparator": None,
        "value": None,
        "unit": None,
        "value_M": None,
        "p_value": None,
        "p_name": None,
        "parse_ok": False,
    }


def parse_pl_index(path: Path) -> pl.DataFrame:
    """Parse `INDEX_general_PL.2020R1.lst`. Returns one row per complex with
    pdb_id, resolution (float or NaN if NMR/unknown), is_nmr, release_year,
    affinity_raw, parsed affinity fields, reference, ligand_code, notes."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line or line.lstrip().startswith("#"):
                continue
            m = _LINE_RE.match(line)
            if not m:
                rows.append({
                    "pdb_id": None, "resolution": None, "is_nmr": None,
                    "release_year": None, "affinity_raw": line,
                    "affinity_type": None, "comparator": None, "value": None,
                    "unit": None, "value_M": None, "p_value": None,
                    "p_name": None, "affinity_parse_ok": False,
                    "reference": None, "ligand_code": None, "notes": None,
                    "raw_line": line,
                })
                continue
            res_raw = m.group("res")
            is_nmr = (res_raw.upper() == "NMR")
            try:
                res = float(res_raw) if not is_nmr else None
            except ValueError:
                res = None
            aff = parse_affinity_string(m.group("aff"))
            rows.append({
                "pdb_id": m.group("pdb").lower(),
                "resolution": res,
                "is_nmr": is_nmr,
                "release_year": int(m.group("year")),
                "affinity_raw": aff["affinity_raw"],
                "affinity_type": aff["affinity_type"],
                "comparator": aff["comparator"],
                "value": aff["value"],
                "unit": aff["unit"],
                "value_M": aff["value_M"],
                "p_value": aff["p_value"],
                "p_name": aff["p_name"],
                "affinity_parse_ok": aff["parse_ok"],
                "reference": m.group("ref"),
                "ligand_code": m.group("lig"),
                "notes": (m.group("notes") or "").strip() or None,
                "raw_line": line,
            })
    return pl.DataFrame(rows)


@dataclass(slots=True)
class ComplexFiles:
    pdb_id: str
    dir: Path
    year_bucket: Optional[str]
    ligand_mol2: Optional[Path]
    ligand_sdf: Optional[Path]
    pocket_pdb: Optional[Path]
    protein_pdb: Optional[Path]


def discover_complexes(extracted_pl_root: Path) -> List[ComplexFiles]:
    """Walk `extracted/P-L/{bucket}/{pdb}/`. Returns one ComplexFiles per pdb dir."""
    out: List[ComplexFiles] = []
    if not extracted_pl_root.exists():
        return out
    for bucket in sorted(p for p in extracted_pl_root.iterdir() if p.is_dir()):
        for cdir in sorted(p for p in bucket.iterdir() if p.is_dir()):
            pdb = cdir.name.lower()
            lig_mol2  = cdir / f"{pdb}_ligand.mol2"
            lig_sdf   = cdir / f"{pdb}_ligand.sdf"
            pocket    = cdir / f"{pdb}_pocket.pdb"
            protein   = cdir / f"{pdb}_protein.pdb"
            out.append(ComplexFiles(
                pdb_id=pdb, dir=cdir, year_bucket=bucket.name,
                ligand_mol2=lig_mol2 if lig_mol2.exists() else None,
                ligand_sdf=lig_sdf if lig_sdf.exists() else None,
                pocket_pdb=pocket if pocket.exists() else None,
                protein_pdb=protein if protein.exists() else None,
            ))
    return out


# -------------------- structure parsers --------------------

def parse_ligand(mol2: Optional[Path], sdf: Optional[Path]):
    """Try MOL2 first (PDBBind tends to provide correct bond orders here);
    fall back to SDF. Returns (canonical_smiles, inchikey, scaffold_smiles,
    n_atoms, source_format, parse_ok)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    mol = None
    fmt = None
    if mol2 and mol2.exists():
        try:
            mol = Chem.MolFromMol2File(str(mol2), sanitize=True, removeHs=True)
            if mol is not None:
                fmt = "mol2"
        except Exception:
            mol = None
    if mol is None and sdf and sdf.exists():
        try:
            suppl = Chem.SDMolSupplier(str(sdf), sanitize=True, removeHs=True)
            for m in suppl:
                if m is not None:
                    mol = m
                    fmt = "sdf"
                    break
        except Exception:
            mol = None
    if mol is None:
        return (None, None, None, 0, fmt, False)
    try:
        canon = Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        canon = None
    try:
        ik = Chem.MolToInchiKey(mol)
    except Exception:
        ik = None
    try:
        scaf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        scaf = None
    return (canon, ik, scaf, mol.GetNumAtoms(), fmt, True)


# Single-letter amino-acid table used when Biopython is not available.
_AA3 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    # tolerated non-canonicals collapse to X
    "MSE":"M","SEC":"U","PYL":"O",
}


def parse_protein_pdb(path: Optional[Path]) -> Dict[str, object]:
    """Minimal protein PDB parsing — no Biopython hard dependency. Walks ATOM
    records, counts CA per chain, builds per-chain sequence by mapping residue
    name to single-letter. Returns dict with chains (sorted list), n_chains,
    n_residues, n_atoms, sequence_by_chain, sequence_concat, parse_ok."""
    info: Dict[str, object] = {
        "chains": [], "n_chains": 0, "n_residues": 0, "n_atoms": 0,
        "sequence_by_chain": {}, "sequence_concat": None, "parse_ok": False,
    }
    if not path or not path.exists():
        return info
    seqs: Dict[str, List[Tuple[int, str]]] = {}
    n_atoms = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                if not raw.startswith("ATOM  "):
                    continue
                n_atoms += 1
                # PDB ATOM record layout (fixed columns).
                atom_name = raw[12:16].strip()
                if atom_name != "CA":
                    continue
                res_name = raw[17:20].strip().upper()
                chain    = raw[21:22] or " "
                try:
                    res_seq = int(raw[22:26].strip())
                except ValueError:
                    continue
                aa = _AA3.get(res_name, "X")
                seqs.setdefault(chain, []).append((res_seq, aa))
    except OSError:
        return info
    seq_by_chain: Dict[str, str] = {}
    n_res = 0
    for c, lst in seqs.items():
        lst.sort(key=lambda r: r[0])
        s = "".join(aa for _, aa in lst)
        seq_by_chain[c] = s
        n_res += len(s)
    info.update({
        "chains": sorted(seq_by_chain.keys()),
        "n_chains": len(seq_by_chain),
        "n_residues": n_res,
        "n_atoms": n_atoms,
        "sequence_by_chain": seq_by_chain,
        "sequence_concat": "|".join(seq_by_chain[c] for c in sorted(seq_by_chain.keys())),
        "parse_ok": (n_res > 0),
    })
    return info


def parse_pdb_atom_count(path: Optional[Path]) -> int:
    """Cheap atom count — ATOM + HETATM lines."""
    if not path or not path.exists():
        return 0
    n = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                if raw.startswith("ATOM  ") or raw.startswith("HETATM"):
                    n += 1
    except OSError:
        return 0
    return n
