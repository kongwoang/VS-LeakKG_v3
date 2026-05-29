# v1 + v2 -> v3 move report

Date: 2026-05-29
Sources:
  - `D:\hoangpc\VS-LeakKG`    (commit `4be0e5f`)   -- v1 flat-schema baseline
  - `D:\hoangpc\VS-LeakKG_v2` (working tree)        -- v2 axis-aligned redesign
Target: `D:\hoangpc\VS-LeakKG_v3`
Mode: **COPY** (v1 and v2 are untouched and remain historical references).

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

### `src/vsleakkg/v2/` -- v2 axis-aligned KG construction (7 files)

Copied from `D:\hoangpc\VS-LeakKG_v2\src\vsleakkg\v2\` into `D:\hoangpc\VS-LeakKG_v3\src\vsleakkg\v2\`.

| File | Role |
|---|---|
| `__init__.py` | v2 subpackage init (`__version__ = "2.0.0-dev"`) |
| `schema.py` | Axis-aligned schema: 13 `NodeType`s incl. `POCKET`, `POCKET_CLUSTER`, `PUBLICATION`, `TRAINSET`; ~20 `EdgeType`s; `DEFAULT_WEIGHTS` (proposal.tex Table 2); `AXIS_EDGE_TYPES` mapping the 7 leakage axes to their edges. **This is the redesign target -- supersedes v1's `graph_schema.py`** |
| `datapaths.py` | Path resolution. ⚠ Currently points at v1 repo as data source (`VSLEAKKG_V1_ROOT` env var, defaults to `D:/hoangpc/VS-LeakKG`). Rework for v3 -- either keep pointing at v1 or rebase onto v3's own `data/` |
| `build_graph.py` | v1->v2 consolidator. `V1_TO_V2_NODE_TYPE` / `V1_TO_V2_EDGE_TYPE` maps; `_map_nodes`, `_map_edges` filter v1 parquets into v2 schema; `_synthesize_pdbbind_examples` collapses Complex->Example and emits `example_has_pocket` |
| `build_side_table.py` | Hydrate-side-table builder. One parquet matching `hydrate.SIDE_TABLE_SCHEMA`: `example_id, source, source_id, smiles, smiles_canonical, inchikey, uniprot, target_sequence, target_sequence_saprot, pdb_id, chembl_id, bindingdb_id, assay_id, label, label_kind`. Bridges v2 `(example_id, partition)` splits to model adapters (SPRINT/DrugCLIP/LigUnity) |
| `hydrate.py` | `make_example_id(source, source_id)` / `parse_example_id`, `SIDE_TABLE_SCHEMA`, `Hydrator` class for O(1) lookup, `canonicalize_smiles` (RDKit-only -- no hash fallback). `KnownSource` enum lists chembl/bindingdb/pdbbind/litpcba/dude/dekois/bayesbind |
| `trainset.py` | Mode B model-specific TrainSet injection: one `TrainSet_m` node + `example_in_trainset` edges per model manifest. Resolves external IDs via id_map; unresolved rows returned (not silently dropped) |

**Imports inside v2/ resolve cleanly within the copied set** -- only intra-package `from .schema`, `from .datapaths`, `from .hydrate`. No dependencies on the skipped downstream modules.

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

Also NOT copied from v1:
- `notebooks/`, `outputs/`, `data/raw/`, `data/processed/` actual contents (all empty in v1 anyway -- datasets are pulled from HF on demand)
- `environments/model_eval_*.yml` (LigUnity / DrugCLIP / HypSeek / Conglude eval configs -- v2/Phase-3 artifacts, not KG construction)
- `_proposal_text.txt`, `cleanup_report.md`, `reproducereport.txt`, `VS_LeakKG.pdf` -- paper drafts and one-off reports

From v2, the following downstream modules were deliberately NOT copied (user is redesigning):

| File | Role in v2 |
|---|---|
| `src/vsleakkg/v2/pipeline.py` | End-to-end Phase 1 orchestrator -- calls leakage_groups + split + validation + baselines |
| `src/vsleakkg/v2/leakage_groups.py` | Union-find / Louvain leakage-group computation per axis |
| `src/vsleakkg/v2/split.py` | Greedy group-atomic split assignment |
| `src/vsleakkg/v2/scoring.py` | -log Dijkstra multi-source contamination scoring + C-NN label transfer |
| `src/vsleakkg/v2/validation_contamination.py` | Three-way contamination matrices (train->test, train->val, val->test) |
| `src/vsleakkg/v2/hubs.py` | Hub-node analysis |
| `src/vsleakkg/v2/label_leakage.py` | Per-row label-leakage attribution |
| `src/vsleakkg/v2/final_figures.py` | Paper-figure renderers |
| `src/vsleakkg/v2/baselines/ligand_only.py` | Morgan-RF ligand-only baseline |
| `src/vsleakkg/v2/baselines/dummy_receptor.py` | Receptor-perturbation control |
| `tools/run_*.py`, `tools/phase*.py`, `tools/consolidate_phase1_report.py` | Phase 1/3 audit suite (~20 tools) |
| `tools/v2_retrieval/build_*_target_kg.py`, `build_*_target_lmdbs.py` | Per-target retrieval KG builders for Phase 2 LigUnity/DrugCLIP eval |
| `scripts/launch_*.sh`, `scripts/phase*.sh` | Phase 1/3 batch launchers |
| `paper.tex`, `proposal.tex`, all `*REPORT*.md` / `PHASE*.md` / `SESSION_HANDOFF.md` | Manuscript drafts and run reports |
| `outputs*/`, `tests/`, `conftest.py`, `docs/`, `mmc2.pdf` | Generated artifacts, tests, docs |
| `pyproject.toml` (v2) | Packaging config -- v3 currently uses plain `requirements.txt` + `environment.yml`; consider adopting later |

## Caveats to be aware of in v3

0. **v3 now has both v1 (flat) and v2 (axis-aligned) schemas side by side.** `src/vsleakkg/*.py` (graph_schema, build_graph, run_*) speaks the v1 schema with prefixed-ID node types (`lig:`, `tgt:`, `complex:`, ...). `src/vsleakkg/v2/*.py` speaks the v2 axis-aligned schema with the 7 leakage axes. The v2 `build_graph.py` consumes v1's processed parquets and produces v2 nodes/edges -- so the pipeline is v1-loaders -> v1-emitter -> v2-consolidator. v3 will probably want to either (a) keep this 2-stage flow but rewrite the v2 consolidator into a from-scratch builder, or (b) collapse the v1 layer and emit v2 nodes/edges directly from the load_*.py files.

1. **`run_overnight.py` is not pure KG construction.** It contains 16 tasks; only `task_6_mvp2_graph` and the chembl/bindingdb loaders are graph-building. Tasks 1-5, 7-16 do contamination scoring, audits, figure generation. You'll likely want to strip those out or split this file when you start v3 modifications.

2. **`run_pdbbind.py` has the same shape.** Its `task_build_graph` and `task_merge_with_mvp` are the KG-construction core; other tasks (`task_smiles_overlap`, `task_target_clustering_*`, `task_report_*`) are audits.

3. **`load_bayesbind.py` was never wired into the KG in v1.** It loads BayesBind into a side table but no `run_*` task emits BayesBind nodes/edges. Decide whether v3 should actually merge BayesBind into the graph.

4. **Pocket schema is partly stubbed.** `Pocket` nodes are emitted by `run_pdbbind.py` with `n_atoms` only; `pocket_similar` and `pocket_in_cluster` edges are NOT generated by any v1 orchestrator (`pocket_cluster.py` would generate them but was never called). If pocket is a v3 priority, this is the gap to fill.

5. **No ChEMBL <-> PDBBind target merging.** `pdbbind_chembl_target_match.py` writes a `pdbbind_chembl_target_matches.parquet` side-table but `task_6_mvp2_graph` never reads it, so `ProteinTarget` and `ChEMBLTarget` remain disconnected in the v1 graph.

6. **Datasets are not in this repo.** `data/raw` and `data/processed` are empty. Run `scripts/fetch_dataset.sh` (needs HF token) to pull `kongwoang/VS_LeakKG :: VS-LeakKG_raw_datasets_20260519.zip`.

7. **v2 `datapaths.py` still points at v1 repo.** It reads `VSLEAKKG_V1_ROOT` env var, defaulting to `D:/hoangpc/VS-LeakKG` (Win) or `~/VS-LeakKG` (Linux). On VUW this means the v2 build_graph expects the v1 clone to also be present at `/vol/dl-nguyenb5-solar/users/hoangpc/VS-LeakKG`. Decide whether v3 should keep this cross-repo dependency or self-host its data.

8. **v2 `build_graph.py` silently drops v1 ChEMBLLigand/ChEMBLTarget/ChEMBLActivity/BindingDBLigand nodes** during v1->v2 mapping (they're not in `V1_TO_V2_NODE_TYPE`). v2 keeps only `ChEMBLAssay -> Assay` and `ChEMBLDocument -> Publication`. If v3 wants ChEMBL/BindingDB to contribute Ligand or Protein nodes in the axis-aligned schema, this map must be extended.

9. **v2 `pocket_similar` and `pocket_in_cluster` edges are NOT generated by any copied module.** Same gap as v1 caveat 4 above. `schema.py` declares the edge types, `build_graph.py` does not emit them. To make the pocket axis feasible for DEKOIS/DUD-E/LIT-PCBA in v3, you need an upstream pocket-clustering or pocket-similarity step (ESM-IF1 embeddings, foldseek, or AA-composition KMeans).

## Git setup applied

```
remote: origin -> https://github.com/kongwoang/VS-LeakKG_v3
branch: main
initial commit: "v3 scaffold: KG construction core inherited from v1"
```
