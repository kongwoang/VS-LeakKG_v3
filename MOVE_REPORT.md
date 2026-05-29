# v1 -> v3 move report

Date: 2026-05-29
Source: `D:\hoangpc\VS-LeakKG` (commit `4be0e5f`)
Target: `D:\hoangpc\VS-LeakKG_v3`
Mode: **COPY** (v1 is untouched and remains a historical reference).

## Files copied INTO v3

### `src/vsleakkg/` — KG construction core (16 files)

| File | Role |
|---|---|
| `__init__.py` | Package init |
| `graph_schema.py` | NodeType / EdgeType enums; prefixed-ID convention (`lig:`, `sca:`, `tgt:`, `prot:`, `complex:`, `pocket:`) |
| `build_graph.py` | `build_examples_frame` + `make_nodes_edges(include_decoy_protocol, include_protein_target)` — per-corpus graph emitter |
| `chem.py` | RDKit canonicalisation, Bemis-Murcko scaffolds, ECFP4 fingerprints |
| `io.py` | Read/write helpers used by all loaders |
| `load_litpcba.py` | LIT-PCBA loader (legacy) |
| `load_litpcba_ave.py` | LIT-PCBA AVE-split loader (primary) |
| `load_dude.py` | DUD-E loader |
| `load_dekois.py` | DEKOIS loader |
| `load_pdbbind.py` | PDBBind loader incl. `parse_affinity_string` for Kd/Ki/IC50 -> pK |
| `load_chembl.py` | ChEMBL flat-file loader |
| `load_chembl_db.py` | ChEMBL SQLite loader |
| `load_bindingdb.py` | BindingDB TSV loader |
| `load_bayesbind.py` | BayesBind loader (loaded but never merged into KG in v1; review for v3) |
| `run_pdbbind.py` | PDBBind orchestrator. `task_build_graph` (lines ~350-475) emits PDBBind-specific node types: `Complex`, `Protein`, `Pocket` (with `n_atoms`), `StructureFile`, `BindingMeasurement`, `AffinityType`. `task_merge_with_mvp` (~493-562) emits cross-source `same_inchikey_as` edges |
| `run_overnight.py` | 16-task MVP-2 orchestrator. `task_6_mvp2_graph` (lines 628-753) emits ChEMBL `Ligand/Target/Activity/Assay/Document` and BindingDB `Ligand` nodes with `chembl_*:` / `bdb_lig:` prefixes |

### `scripts/` — Dataset-fetch infrastructure (9 files)

| File | Role |
|---|---|
| `_dataset_version.ps1` | Pinned: `DatasetHfRepo=kongwoang/VS_LeakKG`, `DatasetZip=VS-LeakKG_raw_datasets_20260519.zip` |
| `dataset_version.sh` | POSIX equivalent of the above |
| `fetch_dataset.ps1`, `fetch_dataset.sh` | Download archive from Hugging Face |
| `extract_datasets.ps1`, `extract_datasets.sh` | Unpack archive into `data/raw` + `data/processed` |
| `download_full_cache.ps1`, `download_full_cache.sh` | Alternative: clone entire HF repo |
| `setup_data.sh` | End-to-end data bootstrap |

### Top-level config

| File | Role |
|---|---|
| `.gitignore` | Carried verbatim — excludes `data/raw/*`, `data/processed/*`, `outputs/*`, `*.parquet`, `*.csv`, `__pycache__`, `.claude/`, etc. |
| `.gitattributes` | Line-ending rules |
| `requirements.txt` | pip deps |
| `environment.yml` | conda env spec |
| `data/MANIFEST.template.md` | Template for documenting fetched dataset versions |
| `data/raw/.gitkeep`, `data/processed/.gitkeep`, `outputs/.gitkeep` | Empty dir placeholders |

## Files NOT copied (deliberately omitted; you said you'll redesign)

These v1 modules implement downstream analysis on top of the KG and are NOT carried into v3:

| File | What it does in v1 |
|---|---|
| `audit_ligand.py` | Ligand-axis identity/scaffold/analog overlap audit |
| `contamination_score.py` | Hop-weighted contamination score `C(x_t)` |
| `weighted_contamination.py` | LIT-PCBA path-product contamination (pocket weight stubbed to 0.0) |
| `contamination_nn.py` | Contamination-nearest-neighbour label-copying baseline (7 axes) |
| `pocket_cluster.py` | Standalone pocket AA-composition KMeans clustering (orphan — never wired into orchestrators) |
| `split_generator.py` | Hash-based group-cold splits at seed=17 for all 7 corpora |
| `split_comparison.py` | Compare paper splits vs cold splits |
| `target_confirmed_provenance.py` | LIT-PCBA candidate-assay provenance |
| `pdbbind_chembl_target_match.py` | UniProt-level PDBBind <-> ChEMBL target matching (side-table only) |
| `pdbbind_cluster_proteins.py` | MMseqs2 30/50/90% protein clustering |
| `metrics.py`, `timebin.py` | Metric helpers, time-binned analysis |
| `decile_worst_group.py`, `source_only_diagnostics.py`, `bayesbind_diagnostics.py`, `diagnostics.py` | Diagnostic CSV/figure generators |
| `final_tables.py`, `final_figures.py`, `final_figures_v2.py` | Paper table/figure renderers |
| `run_mvp_audit.py`, `run_mvp1_audit.py` | Older audit orchestrators superseded by `run_overnight.py` |

Also NOT copied:
- `notebooks/`, `outputs/`, `data/raw/`, `data/processed/` actual contents (all empty in v1 anyway — datasets are pulled from HF on demand)
- `environments/model_eval_*.yml` (LigUnity / DrugCLIP / HypSeek / Conglude eval configs — v2/Phase-3 artifacts, not KG construction)
- `_proposal_text.txt`, `cleanup_report.md`, `reproducereport.txt`, `VS_LeakKG.pdf` — paper drafts and one-off reports

## Caveats to be aware of in v3

1. **`run_overnight.py` is not pure KG construction.** It contains 16 tasks; only `task_6_mvp2_graph` and the chembl/bindingdb loaders are graph-building. Tasks 1-5, 7-16 do contamination scoring, audits, figure generation. You'll likely want to strip those out or split this file when you start v3 modifications.

2. **`run_pdbbind.py` has the same shape.** Its `task_build_graph` and `task_merge_with_mvp` are the KG-construction core; other tasks (`task_smiles_overlap`, `task_target_clustering_*`, `task_report_*`) are audits.

3. **`load_bayesbind.py` was never wired into the KG in v1.** It loads BayesBind into a side table but no `run_*` task emits BayesBind nodes/edges. Decide whether v3 should actually merge BayesBind into the graph.

4. **Pocket schema is partly stubbed.** `Pocket` nodes are emitted by `run_pdbbind.py` with `n_atoms` only; `pocket_similar` and `pocket_in_cluster` edges are NOT generated by any v1 orchestrator (`pocket_cluster.py` would generate them but was never called). If pocket is a v3 priority, this is the gap to fill.

5. **No ChEMBL <-> PDBBind target merging.** `pdbbind_chembl_target_match.py` writes a `pdbbind_chembl_target_matches.parquet` side-table but `task_6_mvp2_graph` never reads it, so `ProteinTarget` and `ChEMBLTarget` remain disconnected in the v1 graph.

6. **Datasets are not in this repo.** `data/raw` and `data/processed` are empty. Run `scripts/fetch_dataset.sh` (needs HF token) to pull `kongwoang/VS_LeakKG :: VS-LeakKG_raw_datasets_20260519.zip`.

## Git setup applied

```
remote: origin -> https://github.com/kongwoang/VS-LeakKG_v3
branch: main
initial commit: "v3 scaffold: KG construction core inherited from v1"
```
