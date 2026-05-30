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
| 5 | Clean old outputs on VUW, rebuild KG, verify | done |

## Phase 5 — final state (2026-05-30 05:42)

### Pipeline runtime
- load_chembl: 0.5s (cached)
- load_bindingdb: 0.3s (cached)
- load_bigbind: 855.6s (~14 min — RDKit canonicalize 583K SMILES single-thread)
- chembl_map: 4.5s — BigBind 99.99% mapped to ChEMBL (confirms BigBind ⊂ ChEMBL)
- bindingdb_map: 3.1s — BigBind 57.78% mapped
- chembl_provenance: 358.8s (~6 min — pulled 7.44M activities for 608K mapped molregnos)
- build_kg: 269.2s (~4.5 min) **before defensive filter** — 16.12M nodes / 60.42M edges with 4.6M corrupted node_ids
- v2/build_graph: 9.6s — produced final v2_nodes/v2_edges

### Bug discovered: polars iter_rows null-byte corruption

`task_build_kg` iterates 7.44M ChEMBL provenance rows via `iter_rows(named=True)` and emits `(f"chembl_{...}:{r['...']}", ...)` tuples. Approximately 4.6M of these emissions produced node_ids filled with **null bytes** (`\x00...` of various lengths 16-24 bytes) instead of the f-string result. Bytes-length distribution matched the lengths of expected node-id prefixes exactly (`chembl_tgt:CHEMBL` = 17 chars → 17 null bytes, etc.), confirming the corruption happens during the f-string interpolation on polars-returned strings.

Workaround applied: defensive filter `nodes.filter(pl.col("node_id").str.contains(":"))` + edge semi-join to prune dangling. Drops 4.86M nodes (4.6M corrupted + 243K post-concat duplicates) and 32M edges.

Root cause guess: polars iter_rows(named=True) on >5M-row DataFrames may return strings that share UCS-2/UCS-4 memory with the underlying Arrow buffer, and rapid f-string allocation interleaves null bytes. Proper fix: convert columns to Python lists via `.to_list()` before the loop. TODO for follow-up commit.

### Final KG (v2 schema)

| Metric | Value |
|---|---:|
| Nodes total | 5,971,854 |
| Edges total | 10,099,165 |
| Example | 3,466,882 |
| Ligand | 1,456,985 |
| Assay | 538,047 |
| Scaffold | 454,197 |
| Publication | 42,537 |
| ProteinCluster | 12,208 |
| Protein | 991 |
| DatasetSource | 5 |
| example_from_source | 3,426,497 |
| example_has_ligand | 2,443,317 |
| example_has_protein | 2,143,548 |
| source_decoy_protocol | 1,091,964 |
| ligand_scaffold | 954,118 |
| protein_in_cluster | 35,586 |
| **ligand_exact** | **3,721** (cross-corpus InChIKey edges — new in v3, was 0 in v2) |
| ligand_similar | 414 |

### Comparison vs. previous v2 build (with PDBBind, with pocket)

| Metric | v2 with PDBBind | v3 with BigBind |
|---|---:|---:|
| Nodes | 6,864,144 | 5,971,854 |
| Edges | 15,714,066 | 10,099,165 |
| Example | 4,181,664 | 3,466,882 |
| Protein | 12,060 | 991 (lost PDBBind anchor; benchmark proteins minimally represented) |
| Pocket | varied | 0 (axis removed) |
| ligand_exact | 0 | 3,721 |

### Known follow-ups (as of v3 KG initial build)

1. **Diagnose & fix iter_rows null-byte bug** in build_kg.task_build_kg. Convert chembl_provenance / mapped_mol / mapped_with_bdb to Python lists before iteration. Re-run build_kg and verify the 4.6M nodes are recovered. **— Fixed in commit 3b4df08.**
2. **Protein-axis anchor**: with PDBBind dropped, only 991 Protein nodes remain. To restore a meaningful protein axis, extract sequences for all benchmark + ChEMBL/BindingDB targets and rebuild MMseqs2 clusters across all of them. **— Still open.**
3. **BayesBind integration**: write `load_bayesbind.py` analogous to `load_bigbind.py` so BayesBind enters the audit as a corpus (currently outside). **— Still open.**
4. **Pocket axis**: hard-removed in v3 redesign. If structure-based audit is wanted, would need re-introducing with proper ESM-IF1 or PocketGen encoder pipeline. **— User confirmed drop in commit 2e20a13 cycle.**

---

## Rebuild from scratch — 2026-05-30 13:05

User instruction: drop pocket axis fully + rebuild clean KG from v3, keep nothing from v2, run merge integrity audit.

### Cleanup performed on VUW
- Removed `pdbbind_*.parquet`, `pdbbind_*.fasta`, `pdbbind_*.tsv` (v2 anchors no longer valid after PDBBind drop)
- Removed `bigbind_*.parquet`, `kg_*.parquet`, `benchmark_to_*.parquet`, `benchmark_chembl_*.parquet` (stale v2 outputs)
- Removed entire `outputs/` tree (logs, reports, v2 graphs)
- Removed `data/raw/DUD-E_pockets_fetched/` (pocket axis dropped)

Kept: ChEMBL/BindingDB raw extracts + per-corpus `*_examples/_nodes/_edges` parquets (these are dataset-loader outputs, not v2 artifacts).

### Bug fixes applied (commit 3b4df08)

1. **iter_rows null-byte fix**: `task_build_kg` now pre-extracts every column it needs from `mapped_mol`/`prov_clean`/`mapped_with_bdb` via `.to_list()` before iterating. This recovered ~5.4M nodes vs the prior build (16.66M total vs 11.27M).
2. **benchmark_lid SMILES selection**: switched from `canonical_smiles_right` (ChEMBL side) to `canonical_smiles` (benchmark side) so cross-references land on the correct per-corpus Ligand node. ChEMBL canonical now stored separately in props as `canonical_smiles_chembl` for traceability.

### Pipeline runtime (rebuild)
- load_chembl: 0.9s (cached)
- load_bindingdb: 1.0s (cached)
- load_bigbind: 14 min (RDKit canonicalize 583K SMILES)
- chembl_map: 3.3s — BigBind 99.99% mapped
- bindingdb_map: 1.6s
- chembl_provenance: 348s — pulled 7.44M activities
- build_kg: 138s
- v2 consolidator: 17s
- merge_audit: 12s
- **Total ~25 min**

### Final KG (clean rebuild)

| Metric | After bug fix |
|---|---:|
| Raw kg_nodes | 16,661,230 |
| Raw kg_edges | 60,665,905 |
| v2 nodes | **8,348,573** |
| v2 edges | **17,765,329** |
| Example | 4,764,621 |
| Ligand | 2,013,247 |
| Scaffold | 645,558 |
| Assay | 857,115 |
| Publication | 66,653 |
| Protein | 1,371 |
| DatasetSource | 6 |
| DecoyProtocol | 2 |
| example_from_source | 4,764,621 |
| example_has_ligand | 4,764,617 |
| example_has_protein | 4,764,621 |
| ligand_scaffold | 1,934,033 |
| source_decoy_protocol | 1,529,687 |
| ligand_exact (cross-corpus) | **6,939** |
| ligand_similar | 811 |
| protein_in_cluster | 0 (clusters intentionally not rebuilt — see TODO #2) |

### Merge integrity audit (outputs/reports/merge_audit_report.md)

**4 invariants — all PASS**:
- Duplicate node_id: 0
- Null-byte node_id: 0 (confirms bug fix)
- Dangling edges by src: 0
- Dangling edges by dst: 0

**Drift cases detected**:

| Case | Count | Risk | Mitigation present? |
|---|---:|---|---|
| 1+2+4: same-InChIKey, different canonical_smiles (tautomer/protonation drift across corpora) | 6,776 InChIKeys → 6,939 excess Ligand nodes | Low (0.34%) | YES — `same_inchikey_as` edges bridge them |
| 3: same parent InChIKey, different full InChIKey (salt/protonation/tautomer) | 178,587 parents have variants (10.1%) | **Medium** | NO — current code does not strip salts. Train-on-HCl-salt / test-on-free-base would underestimate leakage |
| 5: same (source, target, ligand) → multiple Example rows | 324 triples | Trivial | No action |

Case 3 is the only material concern. Fix would require salt-stripping pre-canonicalize in `vsleakkg.chem.featurize` plus emission of `same_parent_inchikey_as` edges. Effort: ~1-2 hours.

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
