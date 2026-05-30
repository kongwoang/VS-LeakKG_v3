# VS-LeakKG v3

Benchmark-integrity knowledge graph for virtual screening. Builds a
contamination-aware provenance graph spanning **5 benchmarks** (LIT-PCBA-AVE,
DUD-E, DEKOIS, BigBind, BayesBind) plus **2 reference databases** (ChEMBL,
BindingDB), exposes it as parquet, and ships a consolidator that produces
the canonical axis-aligned schema used by downstream leakage audits.

## What's in the KG

| Node type | Final count |
|---|---:|
| Example (one per labeled sample) | ~5.0 M |
| Ligand (deduped by canonical SMILES) | ~2.0 M |
| Scaffold (Bemis-Murcko) | ~645 K |
| Protein (UniProt-anchored + per-corpus targets) | ~6.8 K |
| ProteinCluster (MMseqs2 30/50/90 % identity) | ~19 K |
| Assay (ChEMBL + BindingDB) | ~857 K |
| Publication (ChEMBL docs + BindingDB pmid/doi) | ~67 K |
| **Total** | **~8.6 M** |

| Edge type | Count | Purpose |
|---|---:|---|
| `example_has_ligand` / `_protein` / `_from_source` | ~5.0 M each | provenance anchors |
| `ligand_scaffold` | ~1.9 M | scaffold axis |
| `ligand_exact` | 6,939 | same InChIKey, different SMILES (tautomer / stereo) |
| `ligand_parent_exact` | 6,939 | same salt-stripped parent InChIKey |
| `ligand_similar` | varies | Tanimoto ≥ 0.70 over ECFP4 (D5 pipeline) |
| `protein_in_cluster` | 13,506 | MMseqs2 sequence cluster membership |
| `source_decoy_protocol` | ~1.8 M | decoy-protocol provenance |
| `bindingdb_*` (publication, protein, record) | ~800 K | BindingDB enrichment |

Six axes (proposal Section 5.5): **ligand · scaffold · protein · assay · source · time**.
Pocket axis is deliberately not in v3.

## Layout

```
src/vsleakkg/
  build_kg.py              pipeline orchestrator (TASKS list)
  chem.py                  RDKit canonical, scaffolds, InChIKey, salt-strip,
                           parallel featurize
  graph_schema.py          corpus-level NodeType / EdgeType enums
  build_graph.py           corpus-level make_nodes_edges builder helpers
  io.py                    IO helpers
  load_chembl_db.py        ChEMBL SQLite reader (ligands, targets, sequences)
  load_*.py                per-corpus loaders (DEKOIS, DUD-E, LIT-PCBA,
                           LIT-PCBA-AVE, BigBind, BayesBind, BindingDB)
  build_protein_clusters.py extract UniProt sequences from ChEMBL + run
                           MMseqs2 at 30/50/90 % → protein_clusters_*.parquet
  ligand_similarity.py     exact pairwise Tanimoto via bit-bound pruning
  merge_audit.py           5-case merge-integrity audit
  kg/                      canonical schema (axis-aligned)
    schema.py              NodeType, EdgeType, axes, DEFAULT_WEIGHTS
    consolidate.py         raw KG -> canonical (with mapping, hub flag,
                           trivial scaffold drop, cluster edges)

scripts/                   dataset fetch / extract helpers
data/
  raw/                     dataset archives (gitignored)
  processed/               parquet artefacts produced by build_kg
outputs/
  kg/                      canonical KG parquet + stats
  reports/                 build summaries + merge audit report
```

## Pipeline

```
data/raw/{ChEMBL,BindingDB,DEKOIS,DUD-E,LIT-PCBA,BigBind,BayesBind}/
        ↓ (load_*.py + per-corpus loaders)
data/processed/<corpus>_{examples,nodes,edges}.parquet
        ↓ (vsleakkg.build_kg pipeline)
data/processed/kg_nodes.parquet, kg_edges.parquet           ← raw KG
        ↓ (vsleakkg.ligand_similarity   — appends ligand_similar edges)
        ↓ (vsleakkg.build_protein_clusters — writes protein_clusters_*.parquet)
        ↓ (vsleakkg.kg.consolidate)
outputs/kg/canonical_nodes.parquet, canonical_edges.parquet ← audit-ready KG
        ↓
outputs/reports/merge_audit_report.md                       ← integrity check
```

## Quick start

```bash
# 1. Set up Python env (RDKit, polars, mmseqs2)
conda env create -f environment.yml
conda activate vsleakkg

# 2. Pull datasets from Hugging Face (private repo — needs HF token)
bash scripts/fetch_dataset.sh
bash scripts/extract_datasets.sh

# 3. Build the raw KG (~10–15 min on 32 cores)
PYTHONPATH=src python -m vsleakkg.build_kg

# 4. Build protein sequence clusters (~2 min on 32 cores)
PYTHONPATH=src python -m vsleakkg.build_protein_clusters

# 5. Compute exact pairwise ligand similarity edges (~30–45 min on 32 cores)
PYTHONPATH=src python -m vsleakkg.ligand_similarity \
    --kg-nodes data/processed/kg_nodes.parquet \
    --kg-edges data/processed/kg_edges.parquet \
    --threshold 0.70 --workers 32

# 6. Produce the canonical audit-ready KG
mkdir -p outputs/kg
PYTHONPATH=src python -m vsleakkg.kg.consolidate \
    --output-dir outputs/kg --corpus all

# 7. Run the merge-integrity audit
PYTHONPATH=src python -m vsleakkg.merge_audit
cat outputs/reports/merge_audit_report.md
```

## Workflow: Windows ↔ VUW

Code on Windows, compute on VUW. Mirrored via git.

```powershell
# Windows
git push origin main

# VUW
git pull origin main      # or git clone on first time
```

Remote: `https://github.com/kongwoang/VS-LeakKG_v3`.

## Design notes

- **Salt-strip + InChIKey bridges**: same-InChIKey ligands with different
  canonical SMILES (tautomer/stereo) are bridged via `ligand_exact`; salt
  forms via `ligand_parent_exact` (RDKit SaltRemover-stripped parent).
- **Parallel featurize**: `chem.featurize_batch_parallel` and
  `chem.parent_inchikey_batch_parallel` use `multiprocessing.Pool.imap`
  (order-preserving) with length + index-alignment sanity checks. Falls
  back to sequential below `chunksize`.
- **iter_rows null-byte bug workaround**: polars `iter_rows(named=True)` on
  > 5 M-row DataFrames occasionally corrupts strings during f-string
  interpolation. `task_build_kg` pre-extracts every column via `.to_list()`
  before iterating. Build-time invariants (0 duplicate node_id, 0 null-byte
  prefix, 0 dangling edges) catch any regression.
- **Bit-bound pairwise**: `ligand_similarity` uses the Swamidass-Baldi
  popcount window to prune O(N²) brute force into something tractable.
  Symmetric edges are emitted with `src < dst` lexicographically so
  `unique()` collapses duplicates.

## Status

KG construction core: complete. Merge audit: passing 4 invariants.
Downstream (contamination scoring, split generation, baseline reproductions,
paper figures): TODO. See `REDESIGN_LOG.md` for the construction history,
known issues, and the iter_rows bug post-mortem.
