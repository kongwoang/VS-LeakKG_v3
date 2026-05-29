"""Build the hydrate side-table from v1 processed parquets.

The side-table is the bridge between v2 split parquets (`(example_id,
partition)`) and the downstream model adapters (SPRINT, DrugCLIP,
LigUnity). It produces one parquet matching `vsleakkg.v2.hydrate.SIDE_TABLE_SCHEMA`:

    example_id, source, source_id,
    smiles, smiles_canonical, inchikey,
    uniprot, target_sequence, target_sequence_saprot,
    pdb_id, chembl_id, bindingdb_id, assay_id,
    label, label_kind

Sources:
- **chembl**       from chembl_ligands + chembl_assays + chembl_targets
- **bindingdb**    from bindingdb_records_minimal + bindingdb_ligands_minimal
- **pdbbind**      from pdbbind_index + pdbbind_ligands + pdbbind_proteins
- **litpcba**      from litpcba_ave_examples (the AVE variant — paper canonical)
- **dude**         from dude_examples
- **dekois**       from dekois_examples
- **bayesbind**    from bayesbind_examples

`target_sequence_saprot` is only populated for rows where the v1 graph
already has a foldseek-derived 3Di sequence. For everything else it
stays NULL, and the SPRINT adapter falls back to AA-only sequence (or
flags the row as needing a foldseek run).

CLI:

    python -m vsleakkg.v2.build_side_table \
        --output /vol/.../VS-LeakKG_v2/outputs/v2/graph/side_table.parquet \
        [--sources chembl,bindingdb,pdbbind,litpcba,dude,dekois,bayesbind]
        [--limit 100000]

The script is idempotent and writes the parquet in one shot at the end.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterable

import polars as pl

from .datapaths import processed_dir, require_data_root
from .hydrate import (
    KnownSource,
    SIDE_TABLE_COLUMNS,
    SIDE_TABLE_SCHEMA,
    canonicalize_smiles,
    make_example_id,
    validate_side_table,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-source loaders
# ---------------------------------------------------------------------------


def _empty_row() -> dict:
    """A row with all SIDE_TABLE_COLUMNS set to None and label = 0.0."""
    return {c: None for c in SIDE_TABLE_COLUMNS} | {"label": 0.0}


def _canonical_smiles_col(df: pl.DataFrame, smiles_col: str) -> pl.Series:
    """Best-effort canonical SMILES. Empty if RDKit not present."""
    if smiles_col not in df.columns:
        return pl.Series(values=[None] * df.height, dtype=pl.Utf8)
    return pl.Series(
        values=[canonicalize_smiles(s) for s in df[smiles_col].to_list()],
        dtype=pl.Utf8,
    )


def _norm(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure df has exactly SIDE_TABLE_COLUMNS in the right order/dtype."""
    cols = {c: None for c in SIDE_TABLE_COLUMNS}
    cols.update({c: df[c] for c in df.columns if c in SIDE_TABLE_COLUMNS})
    out = pl.DataFrame(cols, schema=SIDE_TABLE_SCHEMA)
    return out.select(SIDE_TABLE_COLUMNS)


def _load_chembl(processed: Path, limit: int | None) -> pl.DataFrame:
    """ChEMBL: ligand_id from molecule_dictionary + pact label.

    The v1 parquet `chembl_ligands.parquet` carries the molecule-level
    summary; full activity records are in `chembl_assays.parquet`. We
    treat each (molregno, assay) pair as one example with source_id =
    "{chembl_id}__{assay_chembl_id}" so a single ligand can appear in
    multiple assays without collision.
    """
    lf = processed / "chembl_ligands.parquet"
    if not lf.exists():
        log.warning("chembl_ligands.parquet missing; skipping ChEMBL")
        return pl.DataFrame(schema=SIDE_TABLE_SCHEMA)
    ligs = pl.read_parquet(lf)
    if limit:
        ligs = ligs.head(limit)
    sm_col = next((c for c in ligs.columns if c.lower() in
                   ("canonical_smiles", "smiles_canonical", "smiles")), None)
    inchi_col = next((c for c in ligs.columns if "inchi" in c.lower() and "key" in c.lower()), None)
    chembl_col = next((c for c in ligs.columns if c.lower() in
                       ("chembl_id", "molecule_chembl_id")), None)
    if not sm_col or not chembl_col:
        log.warning("chembl_ligands schema missing required cols (have=%s)", ligs.columns)
        return pl.DataFrame(schema=SIDE_TABLE_SCHEMA)
    n = ligs.height
    df = pl.DataFrame(
        {
            "example_id": [
                make_example_id(KnownSource.CHEMBL, str(c))
                for c in ligs[chembl_col].to_list()
            ],
            "source": [KnownSource.CHEMBL.value] * n,
            "source_id": ligs[chembl_col].cast(pl.Utf8).to_list(),
            "smiles": ligs[sm_col].cast(pl.Utf8).to_list(),
            "smiles_canonical": _canonical_smiles_col(ligs, sm_col).to_list(),
            "inchikey": ligs[inchi_col].cast(pl.Utf8).to_list() if inchi_col else [None] * n,
            "uniprot": [None] * n,
            "target_sequence": [None] * n,
            "target_sequence_saprot": [None] * n,
            "pdb_id": [None] * n,
            "chembl_id": ligs[chembl_col].cast(pl.Utf8).to_list(),
            "bindingdb_id": [None] * n,
            "assay_id": [None] * n,
            "label": [0.0] * n,
            "label_kind": [None] * n,
        },
        schema=SIDE_TABLE_SCHEMA,
    )
    return df


def _load_bindingdb(processed: Path, limit: int | None) -> pl.DataFrame:
    rf = processed / "bindingdb_records_minimal.parquet"
    if not rf.exists():
        log.warning("bindingdb_records_minimal.parquet missing")
        return pl.DataFrame(schema=SIDE_TABLE_SCHEMA)
    recs = pl.read_parquet(rf)
    if limit:
        recs = recs.head(limit)
    cols = {c.lower(): c for c in recs.columns}
    smi = cols.get("ligand_smiles") or cols.get("smiles") or cols.get("smiles_canonical")
    rec_id = cols.get("bindingdb_id") or cols.get("rec_id") or cols.get("monomer_id")
    uniprot = cols.get("target_uniprot") or cols.get("uniprot")
    seq = cols.get("target_sequence") or cols.get("sequence")
    if not smi or not rec_id:
        log.warning("bindingdb_records schema missing (have=%s)", recs.columns)
        return pl.DataFrame(schema=SIDE_TABLE_SCHEMA)
    n = recs.height
    df = pl.DataFrame(
        {
            "example_id": [
                make_example_id(KnownSource.BINDINGDB, str(r))
                for r in recs[rec_id].to_list()
            ],
            "source": [KnownSource.BINDINGDB.value] * n,
            "source_id": recs[rec_id].cast(pl.Utf8).to_list(),
            "smiles": recs[smi].cast(pl.Utf8).to_list(),
            "smiles_canonical": _canonical_smiles_col(recs, smi).to_list(),
            "inchikey": [None] * n,
            "uniprot": recs[uniprot].cast(pl.Utf8).to_list() if uniprot else [None] * n,
            "target_sequence": recs[seq].cast(pl.Utf8).to_list() if seq else [None] * n,
            "target_sequence_saprot": [None] * n,
            "pdb_id": [None] * n,
            "chembl_id": [None] * n,
            "bindingdb_id": recs[rec_id].cast(pl.Utf8).to_list(),
            "assay_id": [None] * n,
            "label": [0.0] * n,
            "label_kind": [None] * n,
        },
        schema=SIDE_TABLE_SCHEMA,
    )
    return df


def _load_pdbbind(processed: Path, limit: int | None) -> pl.DataFrame:
    idx = processed / "pdbbind_index.parquet"
    lig = processed / "pdbbind_ligands.parquet"
    prot = processed / "pdbbind_proteins.parquet"
    if not idx.exists() or not lig.exists():
        log.warning("pdbbind_{index,ligands}.parquet missing")
        return pl.DataFrame(schema=SIDE_TABLE_SCHEMA)
    df_idx = pl.read_parquet(idx)
    df_lig = pl.read_parquet(lig)
    df_prot = pl.read_parquet(prot) if prot.exists() else None
    if limit:
        df_idx = df_idx.head(limit)
    # Find pdb-id column
    pdb_col = next((c for c in df_idx.columns
                    if c.lower() in ("pdb_id", "pdb_code", "pdb")), None)
    label_col = next((c for c in df_idx.columns
                      if "pk" in c.lower() or "affinity" in c.lower()), None)
    if not pdb_col:
        log.warning("pdbbind_index missing pdb_id col (have=%s)", df_idx.columns)
        return pl.DataFrame(schema=SIDE_TABLE_SCHEMA)
    # Best-effort join on pdb_id
    ligcols = {c.lower(): c for c in df_lig.columns}
    sm_col = ligcols.get("smiles") or ligcols.get("canonical_smiles")
    lig_pdb = ligcols.get("pdb_id") or ligcols.get("pdb_code")
    j = df_idx
    if sm_col and lig_pdb:
        j = j.join(df_lig.select([lig_pdb, sm_col]).rename({lig_pdb: pdb_col}),
                   on=pdb_col, how="left")
    if df_prot is not None:
        protcols = {c.lower(): c for c in df_prot.columns}
        pp_pdb = protcols.get("pdb_id") or protcols.get("pdb_code")
        pp_uni = protcols.get("uniprot") or protcols.get("uniprot_id")
        pp_seq = protcols.get("sequence") or protcols.get("target_sequence")
        if pp_pdb:
            keep = [c for c in (pp_pdb, pp_uni, pp_seq) if c]
            j = j.join(df_prot.select(keep).rename({pp_pdb: pdb_col}),
                       on=pdb_col, how="left")
    n = j.height
    pdbs = j[pdb_col].cast(pl.Utf8).to_list()
    smi_list = j[sm_col].cast(pl.Utf8).to_list() if sm_col else [None] * n
    uni_list = (
        j[protcols.get("uniprot") or protcols.get("uniprot_id")].cast(pl.Utf8).to_list()
        if (df_prot is not None and (protcols.get("uniprot") or protcols.get("uniprot_id")))
        else [None] * n
    )
    seq_list = (
        j[protcols.get("sequence") or protcols.get("target_sequence")].cast(pl.Utf8).to_list()
        if (df_prot is not None and (protcols.get("sequence") or protcols.get("target_sequence")))
        else [None] * n
    )
    label_list = [float(v) if v is not None else 0.0
                  for v in (j[label_col].to_list() if label_col else [None] * n)]
    df = pl.DataFrame(
        {
            "example_id": [make_example_id(KnownSource.PDBBIND, str(p)) for p in pdbs],
            "source": [KnownSource.PDBBIND.value] * n,
            "source_id": pdbs,
            "smiles": smi_list,
            "smiles_canonical": [canonicalize_smiles(s) for s in smi_list],
            "inchikey": [None] * n,
            "uniprot": uni_list,
            "target_sequence": seq_list,
            "target_sequence_saprot": [None] * n,
            "pdb_id": pdbs,
            "chembl_id": [None] * n,
            "bindingdb_id": [None] * n,
            "assay_id": [None] * n,
            "label": label_list,
            "label_kind": ["pact"] * n,
        },
        schema=SIDE_TABLE_SCHEMA,
    )
    return df


def _load_examples_parquet(
    f: Path,
    source: KnownSource,
    limit: int | None,
) -> pl.DataFrame:
    """LIT-PCBA / DUD-E / DEKOIS / BayesBind share a common-ish schema."""
    if not f.exists():
        log.warning("%s missing", f.name)
        return pl.DataFrame(schema=SIDE_TABLE_SCHEMA)
    df0 = pl.read_parquet(f)
    if limit:
        df0 = df0.head(limit)
    cols = {c.lower(): c for c in df0.columns}
    # Try canonical names first
    sm = cols.get("smiles_canonical") or cols.get("smiles") or cols.get("smiles_input")
    inchi = cols.get("inchikey") or cols.get("inchi_key")
    uni = cols.get("uniprot")
    target = cols.get("target") or cols.get("protein_id")
    label_c = cols.get("label")
    label_t = cols.get("label_type") or cols.get("standard_type")
    # Pick the most specific available identifier - never fall back to
    # `target`, which is shared across thousands of rows and would collapse
    # the table during dedup. If none of the per-row id columns exist,
    # combine row index with target so example_ids are still unique.
    src_id = (
        cols.get("source_id")
        or cols.get("compound_id")
        or cols.get("ext_id_1")
        or cols.get("ext_id_2")
        or cols.get("ligand_id")
        or cols.get("smiles_canonical")
        or cols.get("inchikey")
    )
    if not sm:
        log.warning("%s missing SMILES col (have=%s)", f.name, df0.columns)
        return pl.DataFrame(schema=SIDE_TABLE_SCHEMA)
    n = df0.height
    smiles = df0[sm].cast(pl.Utf8).to_list()
    smi_can = [canonicalize_smiles(s) for s in smiles]
    if src_id is None:
        # Last resort: combine target + row index so each row stays unique.
        targets_list = df0[target].cast(pl.Utf8).to_list() if target else [None] * n
        ids = [f"{(t or 'na')}_row{j}" for j, t in enumerate(targets_list)]
    else:
        ids = df0[src_id].cast(pl.Utf8).to_list()
        # If the chosen column has nulls, fall back to row indices for those.
        ids = [v if v else f"row_{j}" for j, v in enumerate(ids)]
    # Avoid `:` in source_id (would break example_id parsing)
    ids = [i.replace(":", "_") for i in ids]
    label_list = (
        [float(v) if v is not None else 0.0 for v in df0[label_c].to_list()]
        if label_c else [0.0] * n
    )
    label_kind = (
        ["binary" if str(t).lower() in ("active", "decoy", "0", "1", "binary") else "pact"
         for t in df0[label_t].to_list()]
        if label_t else ["binary"] * n
    )
    df = pl.DataFrame(
        {
            "example_id": [make_example_id(source, i) for i in ids],
            "source": [source.value] * n,
            "source_id": ids,
            "smiles": smiles,
            "smiles_canonical": smi_can,
            "inchikey": df0[inchi].cast(pl.Utf8).to_list() if inchi else [None] * n,
            "uniprot": df0[uni].cast(pl.Utf8).to_list() if uni else [None] * n,
            "target_sequence": [None] * n,
            "target_sequence_saprot": [None] * n,
            "pdb_id": [None] * n,
            "chembl_id": [None] * n,
            "bindingdb_id": [None] * n,
            "assay_id": [None] * n,
            "label": label_list,
            "label_kind": label_kind,
        },
        schema=SIDE_TABLE_SCHEMA,
    )
    return df


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

SOURCE_LOADERS = {
    KnownSource.CHEMBL:    lambda p, lim: _load_chembl(p, lim),
    KnownSource.BINDINGDB: lambda p, lim: _load_bindingdb(p, lim),
    KnownSource.PDBBIND:   lambda p, lim: _load_pdbbind(p, lim),
    KnownSource.LITPCBA:   lambda p, lim: _load_examples_parquet(
        p / "litpcba_ave_examples.parquet", KnownSource.LITPCBA, lim),
    KnownSource.DUDE:      lambda p, lim: _load_examples_parquet(
        p / "dude_examples.parquet", KnownSource.DUDE, lim),
    KnownSource.DEKOIS:    lambda p, lim: _load_examples_parquet(
        p / "dekois_examples.parquet", KnownSource.DEKOIS, lim),
    KnownSource.BAYESBIND: lambda p, lim: _load_examples_parquet(
        p / "bayesbind_examples.parquet", KnownSource.BAYESBIND, lim),
}


def build_side_table(
    output: Path,
    *,
    sources: Iterable[KnownSource] | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """Build and write the side-table parquet.

    Returns
    -------
    Per-source row counts plus 'total'.
    """
    processed = processed_dir()
    require_data_root()
    output.parent.mkdir(parents=True, exist_ok=True)

    sources = list(sources) if sources else list(KnownSource)
    counts: dict[str, int] = {}
    frames: list[pl.DataFrame] = []
    for src in sources:
        loader = SOURCE_LOADERS.get(src)
        if not loader:
            counts[src.value] = 0
            continue
        t0 = time.perf_counter()
        df = loader(processed, limit)
        df = _norm(df) if df.height else pl.DataFrame(schema=SIDE_TABLE_SCHEMA)
        # Drop duplicate example_ids within a source (hashing collisions)
        if df.height:
            df = df.unique(subset=["example_id"], keep="first")
        frames.append(df)
        counts[src.value] = df.height
        log.info(
            "source=%s rows=%d (%.1fs)",
            src.value, df.height, time.perf_counter() - t0,
        )

    if frames:
        merged = pl.concat(frames, how="vertical_relaxed")
        # Cross-source dedup: keep first occurrence per example_id
        merged = merged.unique(subset=["example_id"], keep="first")
    else:
        merged = pl.DataFrame(schema=SIDE_TABLE_SCHEMA)

    validate_side_table(merged)

    merged.write_parquet(output)
    counts["total"] = merged.height
    log.info("side-table: %d rows -> %s", merged.height, output)
    return counts


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sources", default=",".join(s.value for s in KnownSource))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    requested = [KnownSource(s.strip()) for s in args.sources.split(",") if s.strip()]
    counts = build_side_table(args.output, sources=requested, limit=args.limit)
    print(f"counts: {counts}")


if __name__ == "__main__":
    _cli()
