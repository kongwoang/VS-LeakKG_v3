"""AUROC + bootstrap-CI + Δ-AUROC significance tests.

All routines use numpy for speed; polars is used only for I/O at the
experiment boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def auroc(scores: Iterable[float], labels: Iterable[int]) -> float:
    """Mann–Whitney / Wilcoxon AUROC. NaN if degenerate.

    O(n log n) via rank-sum. No sklearn dependency so this stays cheap.
    """
    s = np.asarray(list(scores), dtype=np.float64)
    y = np.asarray(list(labels), dtype=np.int8)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # Average ties.
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sum_ranks = np.zeros(len(counts), dtype=np.float64)
        for r, idx in zip(ranks, inv):
            sum_ranks[idx] += r
        ranks = sum_ranks[inv] / counts[inv]
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass(frozen=True)
class AUROCwithCI:
    point: float
    lower: float
    upper: float
    n_pos: int
    n_neg: int


def bootstrap_auroc_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> AUROCwithCI:
    """Stratified bootstrap CI for AUROC (resample positives + negatives separately)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    point = auroc(scores, labels)
    if not len(pos_idx) or not len(neg_idx):
        return AUROCwithCI(point, float("nan"), float("nan"), len(pos_idx), len(neg_idx))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        p = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        n = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p, n])
        boots[b] = auroc(scores[idx], labels[idx])
    lo = float(np.nanquantile(boots, alpha / 2))
    hi = float(np.nanquantile(boots, 1 - alpha / 2))
    return AUROCwithCI(point, lo, hi, len(pos_idx), len(neg_idx))


def delta_auroc_test(
    scores_a: np.ndarray, labels_a: np.ndarray,
    scores_b: np.ndarray, labels_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap test for AUROC(a) - AUROC(b).

    Returns dict with `delta`, `ci_low`, `ci_high`, `pvalue_two_sided`.
    Used to compare AUROC on two test partitions (e.g., paper-leak vs clean).
    """
    point_a = auroc(scores_a, labels_a)
    point_b = auroc(scores_b, labels_b)
    delta = point_a - point_b
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    pa = np.flatnonzero(labels_a == 1); na = np.flatnonzero(labels_a == 0)
    pb = np.flatnonzero(labels_b == 1); nb = np.flatnonzero(labels_b == 0)
    if not (len(pa) and len(na) and len(pb) and len(nb)):
        return {"delta": delta, "ci_low": float("nan"), "ci_high": float("nan"),
                "pvalue_two_sided": float("nan"), "point_a": point_a, "point_b": point_b}
    for b in range(n_boot):
        ia = np.concatenate([rng.choice(pa, len(pa), replace=True),
                             rng.choice(na, len(na), replace=True)])
        ib = np.concatenate([rng.choice(pb, len(pb), replace=True),
                             rng.choice(nb, len(nb), replace=True)])
        deltas[b] = auroc(scores_a[ia], labels_a[ia]) - auroc(scores_b[ib], labels_b[ib])
    ci_low = float(np.nanquantile(deltas, 0.025))
    ci_high = float(np.nanquantile(deltas, 0.975))
    # Two-sided p ≈ 2 × min(P(delta<=0), P(delta>=0))
    p_le = float(np.nanmean(deltas <= 0))
    p_ge = float(np.nanmean(deltas >= 0))
    pval = 2 * min(p_le, p_ge)
    return {
        "delta": delta, "point_a": point_a, "point_b": point_b,
        "ci_low": ci_low, "ci_high": ci_high, "pvalue_two_sided": pval,
    }
