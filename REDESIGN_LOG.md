# v3 Redesign: drop PDBBind, add BigBind, remove pocket axis

Started: 2026-05-29 22:30 (user retired for the night with full autonomy)

## Plan (committed sequentially)

| Phase | Goal | Status |
|---|---|---|
| 2a | Drop pocket axis from `v2/schema.py` | pending |
| 2b | Drop PDBBind synthesizer from `v2/build_graph.py` | pending |
| 2c | Delete `run_pdbbind.py` + `load_pdbbind.py` | pending |
| 2d | Rewrite `run_overnight.task_6_mvp2_graph` to read per-corpus parquets directly (mvp1_plus_pdbbind producer is gone) | pending |
| 3a | Write `load_bigbind.py` (parses BigBind activities + structures CSVs) | pending |
| 3b | Add `task_X_bigbind_load` + wire into task_6 | pending |
| 3c | Emit explicit `ligand_exact` cross-corpus InChIKey edges (fill the gap) | pending |
| 4 | Extract BigBind tarball on VUW | pending |
| 5 | Rebuild mvp2 + v2 KG, verify | pending |

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
