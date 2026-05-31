"""Train Morgan-RF on every feasible (corpus, protocol, params, seed=42).

Reads data/splits/audit_summary.csv to identify feasible splits, then
runs experiments.baselines.morgan_rf on each. Skips predictions that
already exist. Writes outputs to:

    data/predictions_v2/<corpus>/morgan_rf__<protocol>__<params>__seed<n>.parquet
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

# Canonical 14 protocol configs (from RESULTS.md).
CANONICAL = {
    ("random", "default"),
    ("random_per_target", "default"),
    ("scaffold", "default"),
    ("scaffold_generic", "default"),
    ("tanimoto_maxmin", "T0.4"),
    ("protein_cluster", "identity30"),
    ("protein_cluster", "identity50"),
    ("protein_cluster", "identity90"),
    ("kg_kdisjoint", "K2_axesligand,scaffold"),
    ("kg_kdisjoint", "K2_axesligand,scaffold,publication,assay"),
    ("kg_maxmin", "T3_axesligand,scaffold"),
    ("kg_maxmin", "T3_axesligand,scaffold,publication,assay"),
    ("kg_axis_budget", "K2_budget_assay1_budget_publication1"),
    ("kg_axis_budget", "K2"),
}


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    audit = pl.read_csv(root / "data/splits/audit_summary.csv").filter(
        pl.col("seed") == 42
    )
    pred_root = root / "data/predictions_v2"
    pred_root.mkdir(parents=True, exist_ok=True)

    corpora = args.corpora.split(",") if args.corpora else ["DEKOIS", "BayesBind", "BigBind", "DUD-E", "LIT-PCBA"]
    queue: list[dict] = []
    for r in audit.iter_rows(named=True):
        if r["corpus"] not in corpora:
            continue
        key = (r["protocol"], r["params"])
        if key not in CANONICAL:
            continue
        if not r["feasible"]:
            continue
        split_path = root / "data/splits" / r["corpus"] / f"{r['protocol']}__{r['params']}__seed{r['seed']}.parquet"
        out_path = pred_root / r["corpus"] / f"morgan_rf__{r['protocol']}__{r['params']}__seed{r['seed']}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            continue
        queue.append({
            "corpus": r["corpus"], "protocol": r["protocol"], "params": r["params"],
            "split_path": str(split_path), "out_path": str(out_path),
        })

    print(f"[{args.tag}] queued {len(queue)} train jobs for corpora={corpora}", flush=True)
    for i, job in enumerate(queue):
        t0 = time.perf_counter()
        cmd = [
            "python3", "-u", "-m", "experiments.baselines.morgan_rf",
            "--split", job["split_path"],
            "--output", job["out_path"],
            "--train-cap", "15000",
        ]
        print(f"[{args.tag}] [{i+1}/{len(queue)}] {job['corpus']} / "
              f"{job['protocol']} / {job['params']} ...", flush=True)
        try:
            subprocess.run(cmd, cwd=str(root), check=True, capture_output=True,
                          text=True, timeout=3600)
            print(f"[{args.tag}]   done ({time.perf_counter() - t0:.1f}s)", flush=True)
        except subprocess.CalledProcessError as ex:
            print(f"[{args.tag}]   FAILED: {ex.returncode}", flush=True)
            print(ex.stderr[-500:], flush=True)
        except subprocess.TimeoutExpired:
            print(f"[{args.tag}]   TIMEOUT", flush=True)


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/vol/dl-nguyenb5-solar/users/hoangpc/VS-LeakKG_v3")
    p.add_argument("--corpora", default="",
                   help="comma-separated subset of corpora (default all)")
    p.add_argument("--tag", default="train",
                   help="prefix for log lines (useful when running parallel)")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
