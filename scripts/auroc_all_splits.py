"""Compute AUROC for every (corpus, protocol) prediction parquet.

Reads data/predictions_v2/<corpus>/morgan_rf__*.parquet, computes
AUROC + 95% bootstrap CI for the test fold. Output: a sortable
table that shows whether KG splits are harder than baselines per
corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path("/vol/dl-nguyenb5-solar/users/hoangpc/VS-LeakKG_v3")
sys.path.insert(0, str(ROOT))
from experiments.common.stats import bootstrap_auroc_ci  # type: ignore


CORPORA = ["DEKOIS", "BayesBind", "BigBind", "DUD-E", "LIT-PCBA"]


def parse_filename(name: str) -> tuple[str, str]:
    """morgan_rf__<protocol>__<params>__seed<n>.parquet → (protocol, params)."""
    stem = name.rsplit(".parquet", 1)[0]
    parts = stem.split("__")
    # stem = morgan_rf, protocol, params, seedN
    if len(parts) != 4 or parts[0] != "morgan_rf":
        return ("?", "?")
    return (parts[1], parts[2])


rows: list[dict] = []
for corpus in CORPORA:
    pred_dir = ROOT / "data/predictions_v2" / corpus
    if not pred_dir.exists():
        continue
    for f in sorted(pred_dir.glob("morgan_rf__*.parquet")):
        protocol, params = parse_filename(f.name)
        df = pl.read_parquet(f).filter(pl.col("fold") == "test")
        n_act = int((df["label"] == 1).sum())
        n_dec = int((df["label"] == 0).sum())
        if n_act < 2 or n_dec < 2:
            continue
        sc = df["score"].to_numpy()
        la = df["label"].to_numpy().astype(int)
        ci = bootstrap_auroc_ci(sc, la, n_boot=500)
        rows.append({
            "corpus": corpus, "protocol": protocol, "params": params,
            "n_test": df.height, "n_active": n_act,
            "auroc": round(ci.point, 4),
            "ci_lo": round(ci.lower, 4), "ci_hi": round(ci.upper, 4),
        })


df = pl.from_dicts(rows)
df.write_csv(ROOT / "data/predictions_v2/morgan_rf_auroc_summary.csv")
print(f"wrote {ROOT}/data/predictions_v2/morgan_rf_auroc_summary.csv ({df.height} rows)")
print()

# Per-corpus delta vs random baseline
print("=== Per-corpus AUROC sorted (random baseline = reference) ===")
for c in CORPORA:
    sub = df.filter(pl.col("corpus") == c).sort("auroc")
    if not sub.height:
        continue
    print(f"\n--- {c} ---")
    random_row = sub.filter((pl.col("protocol") == "random") & (pl.col("params") == "default"))
    rand_auroc = random_row["auroc"][0] if random_row.height else None
    out_lines = []
    for r in sub.iter_rows(named=True):
        delta = r["auroc"] - rand_auroc if rand_auroc else None
        delta_str = f"{delta:+.3f}" if delta is not None else "—"
        flag = " ←KG" if r["protocol"].startswith("kg_") else ""
        out_lines.append(
            f"  {r['protocol']:18s} {r['params']:50s} n={r['n_test']:>6d} act={r['n_active']:>5d}  "
            f"AUROC={r['auroc']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  Δ={delta_str}{flag}"
        )
    print("\n".join(out_lines))
