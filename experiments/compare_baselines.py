"""Compare Morgan-RF vs C-NN AUROC across corpora.

C-NN uses only KG topology (no chemistry). If its AUROC approaches
Morgan-RF on a benchmark, that benchmark's predictability is mostly
contamination, not chemistry.

Outputs outputs/experiments/baseline_compare.csv and report.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from .common import auroc, bootstrap_auroc_ci


CORPORA = ("DEKOIS", "BayesBind", "BigBind", "DUD-E", "LIT-PCBA")
MODELS = ("morgan_rf", "cnn", "morgan_lr")
SPLIT = "random"


def compute_auroc(pred_dir: Path) -> pl.DataFrame:
    rows = []
    for model in MODELS:
        for corpus in CORPORA:
            p = pred_dir / f"{model}__{SPLIT}__{corpus}.parquet"
            if not p.exists():
                continue
            df = pl.read_parquet(p)
            test = df.filter(pl.col("fold") == "test")
            if not test.height:
                continue
            sc = test["score"].to_numpy()
            la = test["label"].to_numpy().astype(int)
            ci = bootstrap_auroc_ci(sc, la, n_boot=500)
            rows.append({
                "model": model,
                "corpus": corpus,
                "split": SPLIT,
                "n_test": test.height,
                "n_pos": int((la == 1).sum()),
                "n_neg": int((la == 0).sum()),
                "auroc": ci.point,
                "ci_lo": ci.lower,
                "ci_hi": ci.upper,
            })
    return pl.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tab = compute_auroc(Path(args.pred_dir))
    tab.write_csv(out / "baseline_compare.csv")
    pivot = tab.pivot(index="corpus", on="model", values="auroc").to_pandas().round(3)
    md = ["# Baseline AUROC comparison (random split)", ""]
    md.append(pivot.to_markdown())
    md.append("")
    md.append("## Per-row CI")
    md.append(tab.to_pandas().round(3).to_markdown(index=False))
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(tab)
    print()
    print("pivot:")
    print(pivot)


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-dir", default="outputs/predictions", type=Path)
    p.add_argument("--output-dir", default="outputs/experiments/baseline_compare", type=Path)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
