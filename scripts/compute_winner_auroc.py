"""Compute AUROC random vs KG-winner per corpus."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path("/vol/dl-nguyenb5-solar/users/hoangpc/VS-LeakKG_v3")
sys.path.insert(0, str(ROOT / "src"))

from experiments.common.stats import auroc, bootstrap_auroc_ci  # type: ignore

CORPORA = ["DEKOIS", "BayesBind", "BigBind", "DUD-E", "LIT-PCBA"]

print(f"{'corpus':10s} {'split':12s} {'n_test':>8s} {'n_active':>8s} {'AUROC':>8s} {'CI_lo':>8s} {'CI_hi':>8s}")
print("-" * 70)
rows: list[dict] = []
for corpus in CORPORA:
    for tag in ["random", "kg_winner"]:
        f = ROOT / f"data/predictions_v2/morgan_rf__{tag}__{corpus}.parquet"
        if not f.exists():
            print(f"{corpus:10s} {tag:12s} MISSING")
            continue
        df = pl.read_parquet(f).filter(pl.col("fold") == "test")
        sc = df["score"].to_numpy()
        la = df["label"].to_numpy().astype(int)
        n_act = int((la == 1).sum())
        n_dec = int((la == 0).sum())
        if n_act < 2 or n_dec < 2:
            print(f"{corpus:10s} {tag:12s} {df.height:>8d} {n_act:>8d} (degenerate)")
            continue
        ci = bootstrap_auroc_ci(sc, la, n_boot=500)
        print(f"{corpus:10s} {tag:12s} {df.height:>8d} {n_act:>8d} "
              f"{ci.point:>8.3f} {ci.lower:>8.3f} {ci.upper:>8.3f}")
        rows.append({"corpus": corpus, "split": tag, "n_test": df.height,
                     "n_active": n_act, "auroc": ci.point,
                     "ci_lo": ci.lower, "ci_hi": ci.upper})

print()
print("=== AUROC delta (KG_winner − random) ===")
for c in CORPORA:
    r = next((x for x in rows if x["corpus"] == c and x["split"] == "random"), None)
    k = next((x for x in rows if x["corpus"] == c and x["split"] == "kg_winner"), None)
    if r and k:
        delta = k["auroc"] - r["auroc"]
        print(f"  {c:10s}  random={r['auroc']:.3f}  kg_winner={k['auroc']:.3f}  Δ={delta:+.3f}")
