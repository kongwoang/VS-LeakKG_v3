"""Shared featurisation for the baseline models.

Morgan/ECFP4 bit-vector fingerprints via RDKit. Falls back to a SMILES-hash
sparse encoding when RDKit is unavailable so unit tests still run.

Ported from `vsleakkg.v2.baselines.ligand_only` with no behavioural changes;
duplicated rather than imported so the v3 tree is self-contained.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def _try_rdkit_fingerprints(
    smiles_list: list[str], n_bits: int = 2048, radius: int = 2
) -> np.ndarray | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.DataStructs import ConvertToNumpyArray
    except ImportError:
        return None
    feats = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        if not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.uint8)
        ConvertToNumpyArray(fp, arr)
        feats[i] = arr
    return feats


def _fallback_features(smiles_list: list[str], n_bits: int = 2048) -> np.ndarray:
    """SMILES-hash sparse encoding for the no-RDKit case."""
    feats = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        if not smi:
            continue
        h = hash(smi) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        idx = rng.integers(0, n_bits, size=16)
        feats[i, idx] = 1
    return feats


def morgan_fingerprints(
    smiles_list: Iterable[str], *, n_bits: int = 2048, radius: int = 2
) -> tuple[np.ndarray, bool]:
    """Featurise SMILES → uint8 (n, n_bits) array. Returns (feats, used_rdkit)."""
    smiles_list = list(smiles_list)
    feats = _try_rdkit_fingerprints(smiles_list, n_bits=n_bits, radius=radius)
    if feats is None:
        return _fallback_features(smiles_list, n_bits=n_bits), False
    return feats, True
