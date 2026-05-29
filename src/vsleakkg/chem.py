"""Chemistry primitives: SMILES canonicalization, InChIKey, ECFP4, Bemis-Murcko,
Tanimoto. RDKit is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

# RDKit emits warnings for sub-MOL2 parse failures and odd valences. Silence them
# for batch processing; callers that need to inspect individual molecules can
# re-enable temporarily.
RDLogger.DisableLog("rdApp.*")

ECFP_RADIUS = 2
ECFP_NBITS = 2048


@dataclass(slots=True)
class MolFeatures:
    """Per-ligand summary produced by `featurize`."""
    smiles_input: str
    smiles_canonical: Optional[str]
    inchikey: Optional[str]
    scaffold_smiles: Optional[str]
    parse_ok: bool


def _parse(smi: str) -> Optional[Chem.Mol]:
    if smi is None:
        return None
    smi = smi.strip()
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    return mol


def canonicalize_smiles(smi: str, isomeric: bool = True) -> Optional[str]:
    mol = _parse(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric)


def inchikey(smi: str) -> Optional[str]:
    mol = _parse(smi)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def bemis_murcko_scaffold(smi: str) -> Optional[str]:
    """Generic Bemis-Murcko scaffold as a canonical SMILES. Empty scaffolds
    (e.g. fully aliphatic acyclic mols) are reported as the empty string."""
    mol = _parse(smi)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf)
    except Exception:
        return None


def ecfp(smi: str, radius: int = ECFP_RADIUS, nbits: int = ECFP_NBITS):
    """RDKit ExplicitBitVect (used by BulkTanimotoSimilarity)."""
    mol = _parse(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def ecfp_bytes(smi: str, radius: int = ECFP_RADIUS, nbits: int = ECFP_NBITS) -> Optional[bytes]:
    """Bit-packed ECFP4 suitable for parquet storage and numpy reload."""
    fp = ecfp(smi, radius, nbits)
    if fp is None:
        return None
    return DataStructs.BitVectToBinaryText(fp)


def bytes_to_fp(b: bytes, nbits: int = ECFP_NBITS):
    # CreateFromBinaryText is a module-level factory; pass the binary blob from
    # BitVectToBinaryText. Bytes are auto-converted to std::string by the C++
    # binding. The `nbits` arg is kept only for API symmetry — RDKit infers it
    # from the binary header.
    return DataStructs.CreateFromBinaryText(b)


def featurize(smi: str) -> MolFeatures:
    """One-shot canonical + scaffold + inchikey. Avoids parsing the same SMILES
    multiple times."""
    mol = _parse(smi)
    if mol is None:
        return MolFeatures(smi, None, None, None, False)
    can = Chem.MolToSmiles(mol, isomericSmiles=True)
    try:
        ik = Chem.MolToInchiKey(mol)
    except Exception:
        ik = None
    try:
        scaf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        scaf = None
    return MolFeatures(smi, can, ik, scaf, True)


def tanimoto(a, b) -> float:
    return DataStructs.TanimotoSimilarity(a, b)


def bulk_tanimoto(query, refs: Sequence) -> np.ndarray:
    """Vectorized Tanimoto from one query against a list of refs."""
    return np.asarray(DataStructs.BulkTanimotoSimilarity(query, list(refs)), dtype=np.float32)


def max_tanimoto_to_set(queries: Sequence, refs: Sequence) -> np.ndarray:
    """For each query, return its max Tanimoto similarity to any ref. None
    fingerprints are skipped (those rows get -1.0)."""
    out = np.full(len(queries), -1.0, dtype=np.float32)
    if not refs:
        return out
    ref_list = list(refs)
    for i, q in enumerate(queries):
        if q is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(q, ref_list)
        out[i] = max(sims) if sims else -1.0
    return out


def count_pairs_above(queries: Sequence, refs: Sequence, thresholds: Iterable[float]) -> dict:
    """Count (query, ref) pairs whose Tanimoto >= each threshold. Uses bulk
    Tanimoto per query; thresholds is iterated cheaply on each row."""
    ths = sorted(set(thresholds))
    counts = {t: 0 for t in ths}
    if not queries or not refs:
        return counts
    ref_list = list(refs)
    for q in queries:
        if q is None:
            continue
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(q, ref_list), dtype=np.float32)
        for t in ths:
            counts[t] += int((sims >= t).sum())
    return counts
