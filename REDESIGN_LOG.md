# v3 Redesign: drop PDBBind, add BigBind, remove pocket axis

Started: 2026-05-29 22:30 (user retired for the night with full autonomy)

## Plan (committed sequentially)

| Phase | Goal | Status |
|---|---|---|
| 2a | Drop pocket axis from `v2/schema.py` | pending |
| 2b | Drop PDBBind synthesizer from `v2/build_graph.py` | pending |
| 2c | Delete `run_pdbbind.py` + `load_pdbbind.py` | pending |
| 2d | Replace `run_overnight.py` with clean `build_kg.py` (KG-only, mvp2→kg rename, audit tasks stripped) | done |
| 3a | Write `load_bigbind.py` (parses BigBind activities + structures CSVs) | done |
| 3b | `build_kg.task_load_bigbind` wired in pipeline; `task_build_kg` rewrites the merge | done |
| 3c | Emit explicit `ligand_exact` cross-corpus InChIKey edges (fill the gap) | done (`cross_src` loop in task_build_kg) |
| 4 | Extract BigBind tarball on VUW (optional — KG only needs metadata CSVs) | pending |
| 5 | Clean old outputs on VUW, rebuild KG, verify | pending |

## Phase 2d + 3a notes

- Renamed `run_overnight.py` → `build_kg.py`. Audit tasks (7, 8, 12, 13, 14, 15) and `task_0_state` deleted entirely. Tasks 10 (BayesBind) and 11 (BigBind metadata stub) replaced by proper `task_load_bigbind` calling new `load_bigbind.build()`.
- `task_build_kg` (was `task_6_mvp2_graph`) no longer reads any `mvp1_*` parquet. It concatenates per-corpus `_nodes/_edges` parquets for litpcba_ave, dude, dekois, bigbind, then layers ChEMBL/BindingDB cross-refs.
- Cross-corpus `same_inchikey_as` edges: previously only emitted by `run_pdbbind.py` (174 total). Now generalised in `task_build_kg`: scan all 4 corpora's `(smi, inchikey)` pairs and emit edges between distinct lig_node_ids that share an InChIKey. Fills the v2 `ligand_exact` gap that was 0 emissions before.
- `load_bigbind.build()` returns `(examples_df, nodes_df, edges_df)` using the shared `vsleakkg.build_graph.build_examples_frame + make_nodes_edges` pipeline so BigBind nodes drop straight into the same dedup namespace (`lig:md5(canonical_smiles)`).
- `v2/build_graph.py` now reads `kg_nodes/kg_edges.parquet` instead of `mvp2_*`.
- BigBind PDBBind-style cluster anchor: v3 inherits the `pdbbind_protein_clusters_*.parquet` (pre-built MMseqs2 outputs). After dropping PDBBind from KG these clusters dangle until we rebuild clustering across all benchmark proteins. Flagged as Phase 6 TODO.

## Key findings before execution

1. v1's `task_6_mvp2_graph` reads `mvp1_plus_pdbbind_nodes/edges.parquet` as base, which we deleted. Producer of `mvp1_nodes.parquet` is NOT present in v3 source (must have been in v1 cli.py or similar). So task_6 needs major rewrite, not just "strip PDBBind".

2. Sanity-checked merge integrity on current v3 KG (before changes):
   - 0 duplicate node_id
   - 16,184 ligands shared across ≥2 benchmark corpora (real leak signal)
   - 123 ligands in all 3 benchmarks
   - Examples never merge across corpora — labels preserved per-source
   - `same_inchikey_as` only 174 edges (all via PDBBind — will be gone after drop)
   - `ligand_exact` (v2 schema declares) — 0 emitted; needs implementation

3. BigBind metadata structure (already inspected):
   - `activities_unfiltered.csv` 1.68M rows
   - `activities_all.csv` 583K rows (filtered)
   - `activities_train.csv` 439K, `activities_test.csv` 108K, `activities_val.csv` 36K
   - `activities_sna_1_{train,test,val}.csv` — alternative split (SNA = same-name activities?)
   - `structures_all.csv` 19,913 rows; structures_{train,test,val}.csv subsets
   - Full archive 18 GB, currently un-extracted in v3 raw

## Edits committed

(populated as phases complete)
