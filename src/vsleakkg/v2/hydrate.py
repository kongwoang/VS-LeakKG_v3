"""Hydrate v2 example IDs into model-consumable rows.

v2 emits `(example_id, partition)` parquets from
`vsleakkg.v2.split.greedy_assign(...)`. Every downstream model adapter
(SPRINT, DrugCLIP, LigUnity) needs richer fields per example: SMILES,
target sequence, UniProt, label, source-specific IDs. This module
provides:

1. The **side-table schema** — a contract for what columns hydrate
   produces. `SIDE_TABLE_SCHEMA` is the single source of truth.
2. **`Hydrator`** — reads a pre-built side-table parquet and serves
   per-example_id rows. Lookup is O(1) after the initial load.
3. **`example_id` helpers** — `make_example_id(source, source_id)` and
   `parse_example_id(eid)`. v2 example IDs are of the form
   `"<source>:<source_id>"` (e.g., `"chembl:ACT_12345"`). Sources are
   one of the values in `KnownSource`.
4. **`canonicalize_smiles(...)`** — RDKit if available, otherwise
   identity. Hashing is intentionally NOT used as a fallback — silently
   producing wrong canonical SMILES is worse than failing fast.
5. **`validate_side_table(...)`** — fail-fast schema check used by both
   the Linux builder (writes the parquet) and the model adapters (read).

The side-table itself is built **on Linux** by `build_graph.py` (not
yet written) because the v1 raw loaders need the unpacked dataset
archive on disk. This module only consumes the parquet.

Typical use from a model adapter:

    >>> from vsleakkg.v2.hydrate import Hydrator
    >>> h = Hydrator.from_parquet("outputs/v2/graph/side_table.parquet")
    >>> rows = h.hydrate(example_ids=["chembl:ACT_1", "pdbbind:1abc"])
    >>> rows.select(["example_id", "smiles_canonical", "uniprot"]).head()
"""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

try:  # polars is in pyproject.toml's required deps
    import polars as pl
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "vsleakkg.v2.hydrate requires polars. "
        "Install with: pip install -e .[dev]"
    ) from exc

try:
    from rdkit import Chem
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


__all__ = [
    "KnownSource",
    "SIDE_TABLE_COLUMNS",
    "SIDE_TABLE_SCHEMA",
    "make_example_id",
    "parse_example_id",
    "canonicalize_smiles",
    "Hydrator",
    "HydrationResult",
    "validate_side_table",
]


class KnownSource(str, enum.Enum):
    """The dataset a row originally came from.

    Used as the prefix of every v2 example_id. New sources may be added
    here without changing the example_id format.
    """

    CHEMBL = "chembl"
    BINDINGDB = "bindingdb"
    PDBBIND = "pdbbind"
    LITPCBA = "litpcba"
    DUDE = "dude"
    DEKOIS = "dekois"
    BAYESBIND = "bayesbind"


SIDE_TABLE_COLUMNS = [
    "example_id",              # canonical "<source>:<source_id>" string
    "source",                  # one of KnownSource values
    "source_id",               # raw source-specific id (without "source:" prefix)
    "smiles",                  # ligand SMILES as recorded by the source
    "smiles_canonical",        # RDKit canonical SMILES (None if RDKit absent)
    "inchikey",                # RDKit InChIKey (None if RDKit absent)
    "uniprot",                 # protein UniProt accession; None for some PDBBind rows
    "target_sequence",         # full protein sequence; None until protein resolution step
    "target_sequence_saprot",  # SaProt-style residue+3Di interleaved; None until foldseek run
    "pdb_id",                  # PDB code if this row anchors a structure; None otherwise
    "chembl_id",               # ChEMBL activity / molecule id (cross-link)
    "bindingdb_id",            # BindingDB row id (cross-link)
    "assay_id",                # cross-source assay id (ChEMBL / BindingDB)
    "label",                   # numeric label: pAct (continuous) or 0/1 (binary)
    "label_kind",              # "pact" | "binary" | None
]

SIDE_TABLE_SCHEMA: "dict[str, pl.DataType]" = {
    "example_id": pl.Utf8,
    "source": pl.Utf8,
    "source_id": pl.Utf8,
    "smiles": pl.Utf8,
    "smiles_canonical": pl.Utf8,
    "inchikey": pl.Utf8,
    "uniprot": pl.Utf8,
    "target_sequence": pl.Utf8,
    "target_sequence_saprot": pl.Utf8,
    "pdb_id": pl.Utf8,
    "chembl_id": pl.Utf8,
    "bindingdb_id": pl.Utf8,
    "assay_id": pl.Utf8,
    "label": pl.Float64,
    "label_kind": pl.Utf8,
}


_EXAMPLE_ID_RE = re.compile(r"^(?P<source>[a-z][a-z0-9_]*):(?P<source_id>[^:].*)$")


def make_example_id(source: "KnownSource | str", source_id: str) -> str:
    """Build a canonical v2 example_id from source + source-specific id."""
    if isinstance(source, KnownSource):
        source_str = source.value
    else:
        source_str = source
    if not source_str or ":" in source_str:
        raise ValueError(f"source must be non-empty and contain no colons, got {source_str!r}")
    if not source_id:
        raise ValueError("source_id must be non-empty")
    if source_id.startswith(":"):
        raise ValueError("source_id must not start with ':'")
    return f"{source_str}:{source_id}"


def parse_example_id(example_id: str) -> "tuple[str, str]":
    """Split `'<source>:<source_id>'` into its parts. Raises on malformed input."""
    m = _EXAMPLE_ID_RE.match(example_id)
    if not m:
        raise ValueError(f"malformed example_id: {example_id!r}")
    return m.group("source"), m.group("source_id")


def canonicalize_smiles(smiles: Optional[str]) -> Optional[str]:
    """Canonical SMILES via RDKit. Returns None if RDKit is unavailable
    or the input cannot be parsed.

    A previous version of this helper fell back to a hash when RDKit
    was missing. That was removed: a hash is not a canonical SMILES,
    and silently returning a wrong canonical form is worse than failing.
    Callers should check for None and skip the row (or install RDKit
    on Linux before building the side-table).
    """
    if smiles is None or not _RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


@dataclass(frozen=True)
class HydrationResult:
    """What `Hydrator.hydrate(...)` returns.

    `rows` always has the SIDE_TABLE_COLUMNS schema. `missing` lists
    requested example_ids that were not in the side-table. Callers
    should treat a non-empty `missing` as a build-time bug and report
    it explicitly rather than silently filtering.
    """

    rows: pl.DataFrame
    missing: list[str]

    @property
    def found(self) -> int:
        return self.rows.height

    @property
    def coverage(self) -> float:
        total = self.found + len(self.missing)
        return self.found / total if total else 1.0


class Hydrator:
    """Reads a pre-built side-table parquet and serves rows by example_id."""

    def __init__(self, side_table: pl.DataFrame):
        validate_side_table(side_table)
        self._side_table = side_table
        # Build an in-memory index for O(1) lookup.
        ids = side_table.get_column("example_id").to_list()
        self._index: dict[str, int] = {eid: i for i, eid in enumerate(ids)}

    @classmethod
    def from_parquet(cls, path: "str | Path") -> "Hydrator":
        df = pl.read_parquet(path)
        return cls(df)

    def hydrate(self, example_ids: Iterable[str]) -> HydrationResult:
        ids = list(example_ids)
        indices = []
        missing = []
        for eid in ids:
            i = self._index.get(eid)
            if i is None:
                missing.append(eid)
            else:
                indices.append(i)
        if indices:
            rows = self._side_table[indices]
        else:
            rows = self._side_table.clear()  # zero-row frame with same schema
        return HydrationResult(rows=rows, missing=missing)

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, example_id: str) -> bool:
        return example_id in self._index


def validate_side_table(df: pl.DataFrame) -> None:
    """Raise ValueError if `df` does not match SIDE_TABLE_SCHEMA exactly.

    Used by both the side-table builder (on Linux) and `Hydrator`. We
    insist on an exact column set + dtype match: silently tolerating
    extra columns invites schema drift across model adapters.
    """
    have = dict(df.schema)
    expected = SIDE_TABLE_SCHEMA
    missing = [c for c in expected if c not in have]
    if missing:
        raise ValueError(f"side-table missing required columns: {missing}")
    extra = [c for c in have if c not in expected]
    if extra:
        raise ValueError(f"side-table has unexpected columns: {extra}")
    wrong = [
        (c, have[c], expected[c]) for c in expected if have[c] != expected[c]
    ]
    if wrong:
        msg = ", ".join(f"{c}: got {h}, expected {e}" for c, h, e in wrong)
        raise ValueError(f"side-table dtype mismatch: {msg}")
    # example_id uniqueness
    if df.height and df["example_id"].n_unique() != df.height:
        n_dup = df.height - df["example_id"].n_unique()
        raise ValueError(f"side-table has {n_dup} duplicate example_id(s)")
    # source must be a KnownSource value
    if df.height:
        known = {s.value for s in KnownSource}
        bad_sources = (
            df.filter(~pl.col("source").is_in(list(known)))
            .get_column("source")
            .unique()
            .to_list()
        )
        if bad_sources:
            raise ValueError(
                f"side-table has unknown source values: {bad_sources}. "
                f"Add to KnownSource if these are legitimate new sources."
            )
