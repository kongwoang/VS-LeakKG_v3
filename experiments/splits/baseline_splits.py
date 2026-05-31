"""Nhóm 1 — Off-the-shelf baseline split protocols.

  random            : stratified random by label
  scaffold          : Bemis-Murcko scaffold partition
  tanimoto_maxmin   : ligand_similar neighbour exclusion (T threshold)
  protein_cluster   : MMseqs2 sequence-cluster partition

All four use the canonical KG's pre-computed edges so no extra preprocessing
(scaffold detection, Tanimoto matrix, MMseqs2 run) is needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..common import edges_of_types
from .base import SplitResult, register_protocol, stratified_random_assign


@register_protocol("random")
def build_random(
    examples: pl.DataFrame,
    kg_dir: Path,
    *,
    seed: int = 42,
    test_ratio: float = 0.15,
    val_ratio: float = 0.0,
) -> SplitResult:
    folds = stratified_random_assign(
        examples, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed
    )
    folds = folds.with_columns(pl.lit(False).alias("leak_mask"))
    return SplitResult(folds=folds, meta={"protocol": "random"})


@register_protocol("random_per_target")
def build_random_per_target(
    examples: pl.DataFrame,
    kg_dir: Path,
    *,
    seed: int = 42,
    test_ratio: float = 0.15,
) -> SplitResult:
    """Per-target stratified random — the canonical DTI baseline.

    Within each target, label-stratified random split. Preserves label
    balance per target and prevents accidentally training on more
    actives from one target than others.
    """
    import numpy as np
    ex_prot = (edges_of_types(["example_has_protein"], kg_dir)
               .select(["src", "dst"])
               .rename({"src": "node_id", "dst": "target"})
               .group_by("node_id")
               .agg(pl.col("target").first()))
    enriched = (examples.select(["node_id", "label"])
                .join(ex_prot, on="node_id", how="left")
                .with_columns(pl.col("target").fill_null("_no_target")))

    rng = np.random.default_rng(seed)
    parts = []
    for grp_keys, g in enriched.group_by(["target", "label"]):
        n = g.height
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = int(round(n * test_ratio))
        fold = np.array(["train"] * n, dtype=object)
        fold[idx[:n_test]] = "test"
        parts.append(g.with_columns(pl.Series("fold", fold)))
    folds = (pl.concat(parts, how="vertical_relaxed")
             .select(["node_id", "fold"])
             .with_columns(pl.lit(False).alias("leak_mask")))
    return SplitResult(folds=folds, meta={"protocol": "random_per_target"})


@register_protocol("scaffold_generic")
def build_scaffold_generic(
    examples: pl.DataFrame,
    kg_dir: Path,
    *,
    seed: int = 42,
    test_ratio: float = 0.15,
) -> SplitResult:
    """Generic Murcko scaffold (framework only — atom types stripped) split.

    Two Bemis-Murcko scaffolds that look different (e.g. benzofuran vs
    quinoline) can map to the same generic Murcko framework once you
    collapse N → C and erase aromaticity tags. Splitting by the generic
    framework is a coarser, stricter scaffold split — used as a
    structural baseline alongside Bemis-Murcko.

    Implementation: hash each BM scaffold SMILES by stripping non-ring
    atom types (a fast proxy for RDKit's MakeScaffoldGeneric). Within
    each generic class, all examples go to the same fold.
    """
    import re
    import numpy as np
    lig_scaf = (edges_of_types(["ligand_scaffold"], kg_dir)
                .select(["src", "dst"])
                .rename({"src": "lig", "dst": "scaffold"}))
    ex_lig = (edges_of_types(["example_has_ligand"], kg_dir)
              .select(["src", "dst"])
              .rename({"src": "node_id", "dst": "lig"}))
    ex_scaf = (ex_lig.join(lig_scaf, on="lig", how="left")
               .group_by("node_id").agg(pl.col("scaffold").first()))

    # Map each Bemis-Murcko scaffold node → generic Murcko framework SMILES
    # via RDKit's MakeScaffoldGeneric (replaces all heavy atoms with carbon
    # and erases bond orders). Falls back to the BM scaffold itself when
    # RDKit can't parse it.
    # Restrict the RDKit compute to scaffolds actually used by THIS corpus's
    # examples — otherwise we'd run MakeScaffoldGeneric on all 645K KG
    # scaffolds, which dominates wall time (~15 min on DEKOIS for what
    # should be a <30s operation).
    from ..common import load_canonical_kg
    used = ex_scaf.select("scaffold").drop_nulls().unique()
    nodes, _ = load_canonical_kg(kg_dir)
    scaf_nodes = (nodes.filter(pl.col("node_type") == "Scaffold")
                  .select(["node_id", "label"])
                  .rename({"node_id": "scaffold", "label": "smi"})
                  .join(used, on="scaffold", how="semi"))

    def _generic(smi: str) -> str:
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            if not smi:
                return ""
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return smi  # fallback
            generic = MurckoScaffold.MakeScaffoldGeneric(mol)
            return Chem.MolToSmiles(generic)
        except Exception:
            return smi

    scaf_nodes = scaf_nodes.with_columns(
        pl.col("smi").map_elements(_generic, return_dtype=pl.Utf8).alias("generic"))
    ex_scaf = (ex_scaf.join(scaf_nodes.select(["scaffold", "generic"]),
                             on="scaffold", how="left"))
    scaffolds = ex_scaf["generic"].drop_nulls().unique().to_list()
    rng = np.random.default_rng(seed)
    rng.shuffle(scaffolds)
    n_test = int(round(len(scaffolds) * test_ratio))
    test_set = set(scaffolds[:n_test])

    folds = (examples.select("node_id")
             .join(ex_scaf, on="node_id", how="left")
             .with_columns(
                 pl.when(pl.col("generic").is_in(list(test_set)))
                 .then(pl.lit("test"))
                 .otherwise(pl.lit("train"))
                 .alias("fold"))
             .select(["node_id", "fold"])
             .with_columns(pl.lit(False).alias("leak_mask")))
    return SplitResult(folds=folds, meta={
        "protocol": "scaffold_generic", "n_classes": len(scaffolds),
        "n_test_classes": n_test,
    })


@register_protocol("scaffold")
def build_scaffold(
    examples: pl.DataFrame,
    kg_dir: Path,
    *,
    seed: int = 42,
    test_ratio: float = 0.15,
) -> SplitResult:
    """Bemis-Murcko scaffold partition via canonical `ligand_scaffold` edges."""
    lig_scaf = (edges_of_types(["ligand_scaffold"], kg_dir)
                .select(["src", "dst"])
                .rename({"src": "lig", "dst": "scaffold"}))
    ex_lig = (edges_of_types(["example_has_ligand"], kg_dir)
              .select(["src", "dst"])
              .rename({"src": "node_id", "dst": "lig"}))
    ex_scaf = (ex_lig.join(lig_scaf, on="lig", how="left")
               .group_by("node_id").agg(pl.col("scaffold").first()))

    scaffolds = ex_scaf["scaffold"].drop_nulls().unique().to_list()
    rng = np.random.default_rng(seed)
    rng.shuffle(scaffolds)
    n_test = int(round(len(scaffolds) * test_ratio))
    test_scafs = set(scaffolds[:n_test])

    folds = (examples.select("node_id")
             .join(ex_scaf, on="node_id", how="left")
             .with_columns(
                 pl.when(pl.col("scaffold").is_in(list(test_scafs)))
                 .then(pl.lit("test"))
                 .otherwise(pl.lit("train"))
                 .alias("fold"))
             .select(["node_id", "fold"]))
    folds = folds.with_columns(pl.lit(False).alias("leak_mask"))
    return SplitResult(folds=folds, meta={
        "protocol": "scaffold", "n_scaffolds": len(scaffolds), "n_test_scaffolds": n_test,
    })


@register_protocol("tanimoto_maxmin")
def build_tanimoto_maxmin(
    examples: pl.DataFrame,
    kg_dir: Path,
    *,
    seed: int = 42,
    test_ratio: float = 0.15,
    T: float = 0.4,
) -> SplitResult:
    """Tanimoto MaxMin: after random init, flip any test item whose ligand is
    a ligand_similar neighbour of a train item back to train.

    `T` is the Tanimoto threshold of the `ligand_similar` edges (already
    computed at T≥0.85 in the canonical KG; this method exploits those
    edges directly). For lower-T variants the canonical sim parquet
    would need to be re-thresholded — out of scope here.
    """
    out = stratified_random_assign(examples, test_ratio=test_ratio, seed=seed)
    sim = edges_of_types(["ligand_similar"], kg_dir).select(["src", "dst"])
    ex_lig = (edges_of_types(["example_has_ligand"], kg_dir)
              .select(["src", "dst"])
              .rename({"src": "node_id", "dst": "lig"}))

    # Flip iteratively until convergence (max 3 passes).
    flipped_total = 0
    for _ in range(3):
        fold_by_lig = (out.join(ex_lig, on="node_id", how="inner")
                       .group_by("lig")
                       .agg(pl.col("fold").first()))
        train_ligs = fold_by_lig.filter(pl.col("fold") == "train").select("lig")
        nbr_to_train = (sim.join(train_ligs.rename({"lig": "src"}), on="src",
                                  how="semi")
                            .select("dst").rename({"dst": "lig"}))
        bad_ligs = nbr_to_train.join(
            fold_by_lig.filter(pl.col("fold") == "test"),
            on="lig", how="semi"
        )
        flip_ex = (ex_lig.join(bad_ligs, on="lig", how="semi")
                   .select("node_id").unique())
        if not flip_ex.height:
            break
        out = out.with_columns(
            pl.when(pl.col("node_id").is_in(flip_ex["node_id"]))
            .then(pl.lit("train"))
            .otherwise(pl.col("fold"))
            .alias("fold"))
        flipped_total += flip_ex.height

    folds = out.with_columns(pl.lit(False).alias("leak_mask"))
    return SplitResult(folds=folds, meta={
        "protocol": "tanimoto_maxmin", "T": T, "flipped": flipped_total,
    })


@register_protocol("protein_cluster")
def build_protein_cluster(
    examples: pl.DataFrame,
    kg_dir: Path,
    *,
    seed: int = 42,
    test_ratio: float = 0.15,
    identity: int = 30,
) -> SplitResult:
    """Assign each Example's protein to its sequence cluster at the given
    identity %, then split clusters into train/test."""
    pic = edges_of_types(["protein_in_cluster"], kg_dir).select(["src", "dst"])
    pic = pic.filter(pl.col("dst").str.contains(f"::{identity}::"))
    ex_prot = (edges_of_types(["example_has_protein"], kg_dir)
               .select(["src", "dst"])
               .rename({"src": "node_id", "dst": "prot"}))
    ex_clust = (ex_prot.join(pic.rename({"src": "prot", "dst": "clust"}),
                              on="prot", how="left")
                .group_by("node_id").agg(pl.col("clust").first()))

    clusters = ex_clust["clust"].drop_nulls().unique().to_list()
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    n_test = max(1, int(round(len(clusters) * test_ratio)))
    test_clusts = set(clusters[:n_test])

    folds = (examples.select("node_id")
             .join(ex_clust, on="node_id", how="left")
             .with_columns(
                 pl.when(pl.col("clust").is_in(list(test_clusts)))
                 .then(pl.lit("test"))
                 .otherwise(pl.lit("train"))
                 .alias("fold"))
             .select(["node_id", "fold"])
             .with_columns(pl.lit(False).alias("leak_mask")))
    return SplitResult(folds=folds, meta={
        "protocol": "protein_cluster", "identity": identity,
        "n_clusters": len(clusters), "n_test_clusters": n_test,
    })
