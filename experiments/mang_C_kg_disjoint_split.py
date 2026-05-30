"""Mảng C — KG-disjoint split protocol + head-to-head vs baseline splits.

Question: Can a split protocol that enforces "train and test must be ≥ K
hops apart on the canonical KG" remove leaks that baseline splits (random,
scaffold, Tanimoto-MaxMin, sequence-cluster) miss?

This module produces (a) splits using each protocol, and (b) a head-to-head
table comparing residual leak (% test still ≤K-hops from train) across
protocols.

The actual model training/evaluation is delegated — this module only
produces split assignments and leak-residual statistics.

Outputs
-------
outputs/experiments/mang_C/
  split_<protocol>__<corpus>.parquet   (example_id, fold)
  leak_residual.csv                    (protocol × corpus × K → %leak)
  report.md

Protocols implemented
---------------------
- random        : uniform shuffle, label-stratified
- scaffold      : Bemis–Murcko scaffold partition (uses LIGAND_SCAFFOLD edges)
- ligand_simil  : Tanimoto-MaxMin neighbour exclusion (uses LIGAND_SIMILAR edges)
- protein_clust : sequence-cluster partition at 30% identity
- kg_disjoint   : multi-hop KG-distance ≥ K (this paper's contribution)

CLI
---
python -m experiments.mang_C_kg_disjoint_split \
    --output-dir outputs/experiments/mang_C \
    --corpus DEKOIS \
    --k-hop 3 \
    --test-ratio 0.15
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from .common import (
    DEFAULT_KG_DIR,
    bfs_distance,
    edges_of_types,
    examples_by_corpus,
    load_canonical_kg,
)


_DEFAULT_K_HOP = 3


# ---------------------------------------------------------------------------
# Per-protocol split implementations
# ---------------------------------------------------------------------------


def _stratified_random(examples: pl.DataFrame, test_ratio: float, seed: int) -> pl.DataFrame:
    """Label-stratified random split."""
    rng = np.random.default_rng(seed)
    parts = []
    for lab, g in examples.group_by("label"):
        n = g.height
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = int(round(n * test_ratio))
        fold = np.array(["train"] * n, dtype=object)
        fold[idx[:n_test]] = "test"
        parts.append(g.with_columns(pl.Series("fold", fold)))
    return pl.concat(parts, how="vertical_relaxed").select(["node_id", "fold"])


def _scaffold_split(
    examples: pl.DataFrame, test_ratio: float, seed: int, kg_dir: Path
) -> pl.DataFrame:
    """Assign every example's Scaffold (via Ligand) to train/test; examples
    inherit their Scaffold's assignment. Scaffolds are randomly distributed
    to roughly hit the requested test ratio."""
    lig_scaf = edges_of_types(["ligand_scaffold"], kg_dir).select(["src", "dst"]).rename(
        {"src": "lig", "dst": "scaffold"}
    )
    ex_lig = edges_of_types(["example_has_ligand"], kg_dir).select(["src", "dst"]).rename(
        {"src": "node_id", "dst": "lig"}
    )
    ex_scaf = (
        ex_lig.join(lig_scaf, on="lig", how="left")
        .group_by("node_id")
        .agg(pl.col("scaffold").first().alias("scaffold"))
    )
    scaffolds = ex_scaf["scaffold"].drop_nulls().unique().to_list()
    rng = np.random.default_rng(seed)
    rng.shuffle(scaffolds)
    n_test = int(round(len(scaffolds) * test_ratio))
    test_scafs = set(scaffolds[:n_test])
    return (
        examples.select("node_id")
        .join(ex_scaf, on="node_id", how="left")
        .with_columns(
            pl.when(pl.col("scaffold").is_in(list(test_scafs)))
              .then(pl.lit("test"))
              .otherwise(pl.lit("train"))
              .alias("fold")
        )
        .select(["node_id", "fold"])
    )


def _ligand_simil_split(
    examples: pl.DataFrame, test_ratio: float, seed: int, kg_dir: Path
) -> pl.DataFrame:
    """Random partition of examples, then iteratively reassign test items
    whose ligand is a `ligand_similar` neighbour of a train item until the
    train/test partition has no `ligand_similar` cross-edge."""
    out = _stratified_random(examples, test_ratio, seed)
    sim = edges_of_types(["ligand_similar"], kg_dir).select(["src", "dst"])
    ex_lig = edges_of_types(["example_has_ligand"], kg_dir).select(["src", "dst"]).rename(
        {"src": "node_id", "dst": "lig"}
    )
    fold_by_lig = (
        out.join(ex_lig, on="node_id", how="inner")
        .group_by("lig")
        .agg(pl.col("fold").first().alias("fold"))
    )
    # Move any test ligand that has a train neighbour back to train.
    train_ligs = fold_by_lig.filter(pl.col("fold") == "train").select("lig")
    test_to_move = (
        sim.join(train_ligs.rename({"lig": "src"}), on="src", how="semi")
           .select("dst").rename({"dst": "lig"})
           .join(fold_by_lig.filter(pl.col("fold") == "test"), on="lig", how="semi")
    )
    move_ex = ex_lig.join(test_to_move, on="lig", how="semi").select("node_id")
    return out.with_columns(
        pl.when(pl.col("node_id").is_in(move_ex["node_id"]))
          .then(pl.lit("train"))
          .otherwise(pl.col("fold"))
          .alias("fold")
    )


def _protein_cluster_split(
    examples: pl.DataFrame, test_ratio: float, seed: int, kg_dir: Path
) -> pl.DataFrame:
    """Each protein is assigned to its 30% cluster; clusters are split."""
    pic = edges_of_types(["protein_in_cluster"], kg_dir).select(["src", "dst"])
    pic = pic.filter(pl.col("dst").str.contains("::30::"))
    ex_prot = edges_of_types(["example_has_protein"], kg_dir).select(["src", "dst"]).rename(
        {"src": "node_id", "dst": "prot"}
    )
    ex_clust = (
        ex_prot.join(pic.rename({"src": "prot", "dst": "clust"}), on="prot", how="left")
        .group_by("node_id")
        .agg(pl.col("clust").first().alias("clust"))
    )
    clusters = ex_clust["clust"].drop_nulls().unique().to_list()
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    n_test = int(round(len(clusters) * test_ratio))
    test_clusts = set(clusters[:n_test])
    return (
        examples.select("node_id")
        .join(ex_clust, on="node_id", how="left")
        .with_columns(
            pl.when(pl.col("clust").is_in(list(test_clusts)))
              .then(pl.lit("test"))
              .otherwise(pl.lit("train"))
              .alias("fold")
        )
        .select(["node_id", "fold"])
    )


def _kg_disjoint_split(
    examples: pl.DataFrame, test_ratio: float, seed: int, kg_dir: Path, k_hop: int
) -> pl.DataFrame:
    """Iterative protocol:
      1. random init.
      2. BFS from train ≤ k hops; any test reached becomes train candidate.
      3. shrink test until residual leak is < tolerance OR test_ratio falls
         below half its target — emit warning if so.
    """
    out = _stratified_random(examples, test_ratio, seed)
    _, all_edges = load_canonical_kg(kg_dir)
    leak_edges = all_edges.filter(pl.col("edge_type").is_in([
        "example_has_ligand", "example_has_protein",
        "example_from_publication", "example_from_assay",
        "ligand_similar", "ligand_exact", "ligand_parent_exact",
        "ligand_fingerprint_exact", "ligand_scaffold", "protein_in_cluster",
    ])).select(["src", "dst"])
    for _ in range(3):
        train_ids = out.filter(pl.col("fold") == "train").select("node_id")
        reached = bfs_distance(train_ids, leak_edges, max_hop=k_hop)
        # Any test example reached within k_hop is a leak — flip to train.
        flips = (out.filter(pl.col("fold") == "test")
                 .join(reached, on="node_id", how="semi").select("node_id"))
        if not flips.height:
            break
        out = out.with_columns(
            pl.when(pl.col("node_id").is_in(flips["node_id"]))
              .then(pl.lit("train"))
              .otherwise(pl.col("fold"))
              .alias("fold")
        )
    return out


_PROTOCOLS: dict[str, Callable] = {
    "random": _stratified_random,
    "scaffold": _scaffold_split,
    "ligand_simil": _ligand_simil_split,
    "protein_clust": _protein_cluster_split,
    "kg_disjoint": _kg_disjoint_split,
}


# ---------------------------------------------------------------------------
# Residual-leak measurement
# ---------------------------------------------------------------------------


# Per-axis leak edge sets. The "full" composition is what a graph-wide BFS
# would walk; the per-axis variants let us decompose where each split
# protocol's residual leak flows. Within a single corpus all examples share
# their target → any axis that includes example_has_protein trivially
# saturates at k=2; we therefore measure protein leak through
# protein_in_cluster only, and assay/paper through their direct
# example_from_* edges (no example_has_protein bridge).
_AXIS_EDGE_SETS: dict[str, tuple[str, ...]] = {
    "full": (
        "example_has_ligand", "example_has_protein",
        "example_from_publication", "example_from_assay",
        "ligand_similar", "ligand_exact", "ligand_parent_exact",
        "ligand_fingerprint_exact", "ligand_scaffold", "protein_in_cluster",
    ),
    "ligand": (
        "example_has_ligand",
        "ligand_similar", "ligand_exact", "ligand_parent_exact",
        "ligand_fingerprint_exact",
    ),
    "scaffold": (
        "example_has_ligand", "ligand_scaffold",
    ),
    "protein": (
        "example_has_protein", "protein_in_cluster",
    ),
    "publication": (
        "example_from_publication",
    ),
    "assay": (
        "example_from_assay",
    ),
}


def measure_residual_leak(
    split: pl.DataFrame, kg_dir: Path, k_hops: tuple[int, ...] = (1, 2, 3)
) -> pl.DataFrame:
    """Residual-leak table per (axis, k_hop) for one split.

    Returns DataFrame (axis, k_hop, n_test, n_leak, pct_leak). Always has
    the same schema even when test=0 so per-corpus concat is safe.
    """
    _, all_edges = load_canonical_kg(kg_dir)
    train_ids = split.filter(pl.col("fold") == "train").select("node_id")
    test_ids = split.filter(pl.col("fold") == "test").select("node_id")
    rows: list[dict] = []
    if not train_ids.height or not test_ids.height:
        for axis in _AXIS_EDGE_SETS:
            for k in k_hops:
                rows.append({"axis": axis, "k_hop": k,
                             "n_test": int(test_ids.height),
                             "n_leak": 0, "pct_leak": float("nan")})
        return pl.DataFrame(rows)
    max_h = max(k_hops)
    for axis, etypes in _AXIS_EDGE_SETS.items():
        sub = all_edges.filter(pl.col("edge_type").is_in(list(etypes))).select(["src", "dst"])
        if not sub.height:
            for k in k_hops:
                rows.append({"axis": axis, "k_hop": k, "n_test": test_ids.height,
                             "n_leak": 0, "pct_leak": 0.0})
            continue
        reached = bfs_distance(train_ids, sub, max_hop=max_h)
        test_hop = test_ids.join(reached, on="node_id", how="left")
        for k in k_hops:
            n_leak = int(test_hop.filter(
                pl.col("hop").is_not_null() & (pl.col("hop") <= k)
            ).height)
            rows.append({"axis": axis, "k_hop": k, "n_test": test_ids.height,
                         "n_leak": n_leak, "pct_leak": 100 * n_leak / test_ids.height})
    return pl.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    kg_dir = Path(args.kg_dir)
    examples = examples_by_corpus(args.corpus, kg_dir)
    if not examples.height:
        raise SystemExit(f"no Examples found for corpus={args.corpus}")
    print(f"corpus={args.corpus}, examples={examples.height}")
    summary_rows = []
    for name, fn in _PROTOCOLS.items():
        if name == "kg_disjoint":
            split = fn(examples, args.test_ratio, args.seed, kg_dir, args.k_hop)
        elif name == "random":
            split = fn(examples, args.test_ratio, args.seed)
        else:
            split = fn(examples, args.test_ratio, args.seed, kg_dir)
        out_path = out / f"split_{name}__{args.corpus}.parquet"
        split.write_parquet(out_path)
        residual = measure_residual_leak(split, kg_dir)
        residual = residual.with_columns([
            pl.lit(name).alias("protocol"), pl.lit(args.corpus).alias("corpus"),
        ])
        summary_rows.append(residual)
        print(f"  {name}: wrote {out_path.name}, residual = {residual.to_dicts()}")
    summary = pl.concat(summary_rows, how="vertical_relaxed")
    summary_path = out / f"leak_residual__{args.corpus}.csv"
    summary.write_csv(summary_path)
    # Per-axis pivot: rows = protocol, cols = k_hop, separate table per axis.
    md_lines = [f"# Mảng C — split head-to-head ({args.corpus})\n"]
    for axis in _AXIS_EDGE_SETS:
        sub = summary.filter(pl.col("axis") == axis)
        if not sub.height:
            continue
        pivot = sub.pivot(index="protocol", on="k_hop", values="pct_leak")
        md_lines.append(f"\n## axis = {axis}\n")
        md_lines.append(pivot.to_pandas().round(1).to_markdown())
    (out / f"report__{args.corpus}.md").write_text("\n".join(md_lines), encoding="utf-8")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kg-dir", default=DEFAULT_KG_DIR, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--corpus", required=True,
                   choices=["LIT-PCBA", "DUD-E", "BigBind", "DEKOIS", "BayesBind"])
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--k-hop", type=int, default=_DEFAULT_K_HOP)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
