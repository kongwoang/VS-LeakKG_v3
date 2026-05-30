"""Merge integrity audit for the canonical KG.

Detects 5 cases where "same chemical / structural entity" could end up as
two distinct nodes due to ID-computation drift:

  1. SMILES canonicalization drift
       Two corpora produce different canonical_smiles for the same molecule
       (same InChIKey). They get different lig:md5(canonical_smiles) IDs.
  2. ChEMBL canonical_smiles != benchmark canonical_smiles
       The benchmark_lid in task_build_kg is computed from one side; if the
       other side has a Ligand node with a different canonical, you get a
       split lig:* node pair.
  3. Salt-stripping inconsistency
       Same parent molecule, one corpus ships HCl salt form, another ships
       free base. Different canonical_smiles, different InChIKey usually.
  4. Tautomer / stereo encoding difference
       Same InChIKey-first-block but different InChIKey-full or different
       canonical_smiles.
  5. Within-corpus row-index ghosts
       Same (source, target, ligand_node_id) appears in multiple Example
       rows because example_id depends on row_idx — a known harmless dup
       within a corpus, but worth reporting.

Run:
    PYTHONPATH=src python -m vsleakkg.merge_audit
Outputs:
    outputs/reports/merge_audit_report.md
"""
from __future__ import annotations
import json
import logging
from collections import defaultdict
from pathlib import Path

import polars as pl

log = logging.getLogger("vsleakkg.merge_audit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def main() -> int:
    out_lines: list[str] = []
    out_lines.append("# Merge integrity audit\n")
    out_lines.append("Run on rebuilt v3 KG.\n")

    # ---------- Case 1+2+4: SMILES drift via InChIKey collision ----------
    log.info("Case 1+2+4: cross-corpus same-InChIKey, different-SMILES check")
    parts = []
    for corpus, fname in [
        ("LIT-PCBA-AVE", "litpcba_ave_examples.parquet"),
        ("DUD-E",        "dude_examples.parquet"),
        ("DEKOIS",       "dekois_examples.parquet"),
        ("BigBind",      "bigbind_examples.parquet"),
        ("BayesBind",    "bayesbind_examples.parquet"),
    ]:
        p = PROCESSED / fname
        if not p.exists():
            continue
        cols = pl.read_parquet_schema(p)
        smi_col = "smiles_canonical" if "smiles_canonical" in cols else ("canonical_smiles" if "canonical_smiles" in cols else None)
        if smi_col is None or "inchikey" not in cols:
            continue
        df = (pl.scan_parquet(p)
              .select([pl.col(smi_col).alias("smi"), pl.col("inchikey")])
              .filter(pl.col("smi").is_not_null() & pl.col("inchikey").is_not_null())
              .unique()
              .with_columns(pl.lit(corpus).alias("corpus"))
              .collect())
        parts.append(df)
    if parts:
        allsm = pl.concat(parts, how="vertical_relaxed")
        # Group by inchikey -> distinct SMILES count
        g = allsm.group_by("inchikey").agg([
            pl.col("smi").n_unique().alias("n_distinct_smi"),
            pl.col("smi").unique().alias("smis"),
            pl.col("corpus").unique().alias("corpora"),
        ])
        drift = g.filter(pl.col("n_distinct_smi") > 1)
        out_lines.append("\n## Case 1+2+4: cross-corpus same-InChIKey, different canonical SMILES\n")
        out_lines.append(f"- Total unique InChIKeys: {g.height:,}\n")
        out_lines.append(f"- InChIKeys with >1 canonical SMILES (drift): **{drift.height:,}**\n")
        if drift.height:
            out_lines.append("- Top 5 examples:\n")
            top = drift.sort("n_distinct_smi", descending=True).head(5)
            for r in top.iter_rows(named=True):
                out_lines.append(f"  - `{r['inchikey']}`: {r['n_distinct_smi']} SMILES across corpora {r['corpora']}\n")
                for s in r['smis'][:3]:
                    out_lines.append(f"    - `{s[:80]}`\n")
        # By corpus pair: how many InChIKeys differ
        if drift.height:
            pair_counts: dict[tuple, int] = defaultdict(int)
            for r in drift.iter_rows(named=True):
                cs = tuple(sorted(r['corpora']))
                pair_counts[cs] += 1
            out_lines.append("\n  Pair breakdown (corpora that disagree):\n")
            for cs, n in sorted(pair_counts.items(), key=lambda x: -x[1])[:10]:
                out_lines.append(f"    - {'+'.join(cs)}: {n} InChIKeys\n")

    # ---------- Case 3: Salt-stripping (InChIKey first block) ----------
    log.info("Case 3: same parent InChIKey-first-block, different full InChIKey")
    if parts:
        with_parent = allsm.with_columns(
            pl.col("inchikey").str.head(14).alias("parent_ik")
        )
        g2 = with_parent.group_by("parent_ik").agg([
            pl.col("inchikey").n_unique().alias("n_full_ik"),
            pl.col("inchikey").unique().alias("full_iks"),
            pl.col("corpus").unique().alias("corpora"),
        ])
        salt = g2.filter(pl.col("n_full_ik") > 1)
        out_lines.append("\n## Case 3: same parent InChIKey-first-block, different full InChIKey\n")
        out_lines.append(f"- Unique parent InChIKey blocks: {g2.height:,}\n")
        out_lines.append(f"- Parents with >1 full-InChIKey variant: **{salt.height:,}**\n")
        out_lines.append("- Likely causes: salt/no-salt, protonation state, or tautomer differences across corpora.\n")
        if salt.height:
            out_lines.append("- Top 3 examples:\n")
            top = salt.sort("n_full_ik", descending=True).head(3)
            for r in top.iter_rows(named=True):
                out_lines.append(f"  - parent `{r['parent_ik']}`: {r['n_full_ik']} variants across {r['corpora']}\n")

    # ---------- Cross-corpus Ligand node duplication count ----------
    log.info("Case 1+2+4 quantified at KG-node level")
    kg_n_path = PROCESSED / "kg_nodes.parquet"
    if kg_n_path.exists():
        kg = pl.read_parquet(kg_n_path)
        lig_nodes = kg.filter(pl.col("node_type") == "Ligand")
        out_lines.append("\n## Case 1+2+4 at KG-node level\n")
        out_lines.append(f"- Total Ligand nodes in KG: {lig_nodes.height:,}\n")
        # Extract canonical_smiles back: it's stored as the `label` column
        if lig_nodes.height:
            # Distinct InChIKeys covered by these Ligand nodes — we need to derive
            # inchikey from the canonical_smiles to estimate redundancy
            # but that requires RDKit; instead count nodes vs unique InChIKey across corpora.
            ik_counts = allsm["inchikey"].n_unique() if parts else None
            if ik_counts:
                redundancy = lig_nodes.height - ik_counts
                out_lines.append(f"- Distinct InChIKeys across corpora: {ik_counts:,}\n")
                out_lines.append(f"- Excess Ligand nodes (= same InChIKey, different SMILES): **{max(0, redundancy):,}**\n")
                out_lines.append(f"- This number SHOULD equal the per-corpus drift Case 1+2+4 count.\n")

    # ---------- Case 5: Row-idx ghosts (same source+target+ligand, multiple Example rows) ----------
    log.info("Case 5: within-corpus row-idx ghosts")
    out_lines.append("\n## Case 5: within-corpus (source, target, ligand) ghosts\n")
    kg_e_path = PROCESSED / "kg_edges.parquet"
    if kg_n_path.exists() and kg_e_path.exists():
        edges = pl.read_parquet(kg_e_path)
        ex_lig = (edges.filter(pl.col("edge_type") == "example_has_ligand")
                  .select(["src", "dst"])
                  .rename({"src": "example_id", "dst": "ligand_id"}))
        ex_prot = (edges.filter(pl.col("edge_type") == "example_targets_protein")
                   .select(["src", "dst"])
                   .rename({"src": "example_id", "dst": "target_id"}))
        ex_src = (edges.filter(pl.col("edge_type") == "example_from_source")
                  .select(["src", "dst"])
                  .rename({"src": "example_id", "dst": "source_id"}))
        joined = ex_lig.join(ex_prot, on="example_id", how="inner").join(ex_src, on="example_id", how="inner")
        ghost = (joined.group_by(["source_id", "target_id", "ligand_id"]).len()
                 .rename({"len": "n_examples"})
                 .filter(pl.col("n_examples") > 1))
        out_lines.append(f"- Triples (source, target, ligand) with >1 Example: **{ghost.height:,}**\n")
        if ghost.height:
            by_n = (ghost.group_by("n_examples").len()
                    .rename({"len": "n_triples"})
                    .sort("n_examples", descending=True))
            out_lines.append("- Distribution of duplication count:\n")
            for r in by_n.head(5).iter_rows(named=True):
                out_lines.append(f"  - {r['n_examples']} Examples per (s,t,l): {r['n_triples']:,} triples\n")

    # ---------- Final node integrity ----------
    log.info("Final node/edge integrity sanity")
    if kg_n_path.exists():
        kg = pl.read_parquet(kg_n_path)
        e = pl.read_parquet(kg_e_path) if kg_e_path.exists() else pl.DataFrame()
        out_lines.append("\n## Final invariants\n")
        out_lines.append(f"- Total nodes: {kg.height:,}\n")
        out_lines.append(f"- Total edges: {e.height:,}\n")
        dup_id = kg.group_by("node_id").len().filter(pl.col("len") > 1)
        out_lines.append(f"- Duplicate node_id: {dup_id.height} (should be 0)\n")
        null_byte = kg.filter(~pl.col("node_id").str.contains(":"))
        out_lines.append(f"- Node IDs missing `:` prefix (corruption marker): {null_byte.height} (should be 0)\n")
        if e.height:
            ids = kg.select("node_id")
            d_src = e.join(ids.rename({"node_id": "src"}), on="src", how="anti").height
            d_dst = e.join(ids.rename({"node_id": "dst"}), on="dst", how="anti").height
            out_lines.append(f"- Dangling edges by src: {d_src} (should be 0)\n")
            out_lines.append(f"- Dangling edges by dst: {d_dst} (should be 0)\n")

    out_path = REPORTS / "merge_audit_report.md"
    out_path.write_text("".join(out_lines), encoding="utf-8")
    log.info("wrote %s", out_path)
    print(f"WROTE {out_path}")
    return 0


if __name__ == "__main__":
    main()
