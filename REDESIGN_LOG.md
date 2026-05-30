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

## Phase A-E (2026-05-30 13:30 onwards) — feature complete

### Renames (commit 61ef274)
- `src/vsleakkg/v2/` → `src/vsleakkg/kg/` (canonical KG schema layer)
- `v2/build_graph.py` → `kg/consolidate.py` (function `build_graph()` → `consolidate()`)
- `V1_TO_V2_NODE_TYPE` → `CORPUS_TO_CANONICAL_NODE_TYPE` (and EDGE_TYPE)
- `V1_DROPPED_*` → `DROPPED_*`
- `v2_nodes/v2_edges.parquet` → `canonical_nodes/canonical_edges.parquet`
- `outputs/v2/graph/` → `outputs/kg/`
- `VSLEAKKG_V1_ROOT` env → `VSLEAKKG_ROOT`
- All "v1 / v2 / mvp" wording stripped from docstrings.

### A1 + A6: salt-strip parent edge + build-time invariants
- `chem.parent_inchikey`: RDKit SaltRemover → InChIKey of parent. `task_build_kg`
  emits `same_parent_inchikey_as` edges bridging salt/protonation variants
  (in this dataset, no salts were found — 6,939 edges, same as
  `same_inchikey_as`). EdgeType `LIGAND_PARENT_EXACT` added (weight 0.95).
- `task_build_kg` raises on any of: duplicate node_id after dedup, node_id
  missing `:` prefix, dangling edges. Catches the iter_rows regression at
  write time.

### A3: BayesBind wired
- `load_bayesbind.build()`: parses per-target actives.csv + random.csv,
  uses shared `make_nodes_edges`. 261K examples, 21K ligands, 25 targets.
- BayesBind also mapped 99.98% to ChEMBL (confirms BayesBind ⊂ ChEMBL).

### A4: BindingDB enrichment
- `task_build_kg` reads `bindingdb_records_minimal.parquet` for every
  mapped BindingDB ligand and emits Publication (PMID/DOI), Protein
  (UniProt), and Assay-like (record_id) nodes. ~26K Publications,
  ~5K Proteins, ~785K Assays added.

### A* parallel featurize
- `chem.featurize_batch_parallel` / `chem.parent_inchikey_batch_parallel`:
  order-preserving multiprocessing with index-alignment sanity checks.
- Empirical speedup on VUW (32 cores):

  | step | sequential | parallel | speedup |
  |---|---:|---:|---:|
  | BigBind featurize 583K | 14 min | (cached) | n/a |
  | BayesBind featurize 261K | ~6 min | **29.6 s** | 12× |
  | parent_inchikey 2.01M | ~17 min | **2 min 13 s** | 7.7× |

### B2: protein clustering rebuilt
- `load_chembl_db.load_target_sequences`: joins target_dictionary →
  target_components → component_sequences for every protein-type
  component with non-null sequence.
- `build_protein_clusters` orchestrator: extracts 4,696 unique ChEMBL
  sequences, writes FASTA, runs `mmseqs easy-cluster` at 30/50/90 %
  identity, converts each `*_cluster.tsv` to
  `protein_clusters_{30,50,90}.parquet` (schema: accession, cluster_id,
  resolution). MMseqs2 itself uses all 32 cores natively.
- `kg.consolidate` reads the new parquets, prefixes accessions with
  `protein:` to match the KG Protein ID format, then prunes any edges
  whose src lacks a matching Protein node. Net result: 13,506 valid
  `protein_in_cluster` edges, 19,171 ProteinCluster nodes.

### D5: exact pairwise ligand similarity
- `ligand_similarity.py`: ECFP4 fingerprints via parallel pool, sort by
  popcount, bit-bound pruning via the Swamidass-Baldi inequality
  (`Tanimoto(A,B) ≤ min(|A|,|B|)/max(|A|,|B|)`), per-window
  `BulkTanimotoSimilarity`.
- Fingerprint phase: 2.01M ligands → 49.2 s on 32 cores.
- Pairwise phase: ~30 min wall clock for threshold = 0.70.
- Emits each pair with `src < dst` so `unique()` collapses duplicates.

### Final canonical KG (post Phase A-E rebuild)

| Metric | Value |
|---|---:|
| Raw kg | 17.74M nodes / 64.58M edges |
| Canonical KG | **8.63M nodes / 18.83M edges** |
| Example | 5,025,497 (+260K BayesBind) |
| Ligand | 2,013,247 |
| Scaffold | 645,558 |
| Assay | 857,115 |
| Publication | 66,653 |
| Protein | 6,764 (vs 1,371 before — +5,343 from BindingDB UniProt) |
| ProteinCluster | 19,171 (NEW) |
| `ligand_exact` | 6,939 |
| `ligand_parent_exact` | 6,939 (NEW) |
| `ligand_similar` (Tanimoto ≥ 0.70) | TBD (D5 in progress) |
| `protein_in_cluster` | 13,506 (NEW) |

### Tests
- `tests/test_chem.py`: 10 unit tests for canonical SMILES / InChIKey /
  scaffold determinism, salt-strip parent equivalence, and parallel batch
  order preservation. All pass on current pinned RDKit version.

---

## Final D5 result (2026-05-30 21:26)

- Sim wall clock: 14,405s (~4 hours) on 16 workers due to CPU contention
  with another user's process (load avg ~50 throughout). Estimate at full
  speed would have been ~45 min.
- 762,842 ligand_similar pairs at Tanimoto ≥ 0.85 (ECFP4, 2048-bit).
- After `unique()` dedup at consolidate time: **763,653** ligand_similar
  edges (slight bump because some pairs also surfaced from per-corpus
  loader's `ligand_similar_to_ligand`).
- One bug discovered + fixed in `kg/consolidate.py`
  `CORPUS_TO_CANONICAL_EDGE_TYPE`: the table only mapped
  `ligand_similar_to_ligand` from per-corpus loaders, so the 762K
  `ligand_similar` edges emitted directly by `ligand_similarity.py` were
  being filtered out by `_map_edges`. Added an identity entry for
  `ligand_similar -> LIGAND_SIMILAR` so both formats survive.

### Final canonical KG (post D5)

| Metric | Value |
|---|---:|
| Nodes | 8,634,015 |
| Edges | 19,592,120 |
| Example | 5,025,497 |
| Ligand | 2,013,247 |
| Scaffold | 645,558 |
| Assay | 857,115 |
| Publication | 66,653 |
| Protein | 6,764 |
| ProteinCluster | 19,171 |
| example_has_ligand | 5,025,493 |
| example_has_protein | 5,025,497 |
| example_from_source | 5,025,497 |
| ligand_scaffold | 1,934,033 |
| source_decoy_protocol | 1,790,563 |
| **ligand_similar (T ≥ 0.85)** | **763,653** |
| ligand_exact | 6,939 |
| ligand_parent_exact | 6,939 |
| protein_in_cluster | 13,506 |

### Merge audit (raw KG, post D5)
- Raw kg_nodes: 17,739,452, kg_edges: 65,346,527
- 4 invariants pass: 0 duplicate node_id, 0 null-byte, 0 dangling src/dst
- Drift cases (unchanged from pre-D5 audit):
  - 6,776 InChIKey-level (tautomer/protonation, bridged via `ligand_exact`)
  - 178,587 parent-skeleton (salt/protonation/stereo, partially bridged via
    `ligand_parent_exact`)
  - 324 within-corpus ghost triples (trivial)

## Post-anomaly fixes (2026-05-30 23:00)

### Anomalies surfaced by the final sanity check
1. 314,683 `ligand_similar` pairs at Tanimoto = 1.0 — same molecule modulo
   stereo / tautomer that ECFP4 (radius=2) doesn't encode. Audit was
   underweighting these as similarity (0.65) when they should be near-
   identity (0.95).
2. 841 orphan Protein + 10,442 orphan ProteinCluster nodes (degree 0
   after dangling-edge prune).
3. 453 `ligand_similar` rows whose Tanimoto < 0.85 — turned out to be
   per-corpus loader edges with threshold 0.81-0.84, not a parse failure.
   No action.

### Fixes
- New `EdgeType.LIGAND_FINGERPRINT_EXACT` (`ligand_fingerprint_exact`),
  weight 0.95, in the "ligand" axis. `ligand_similarity.py` emits this
  edge type when Tanimoto >= 0.9995; `ligand_similar` for T < 0.9995.
  Re-labeled the existing 314,683 edges in place (no recompute).
- `kg.consolidate`: drop orphan Protein and ProteinCluster nodes via a
  semi-join (switched from `is_in(Series)` which polars 1.x had
  deprecated to ambiguous behaviour and silently kept all orphans).

### Final canonical KG (post fixes)

| Metric | Value |
|---|---:|
| Nodes | **8,622,732** |
| Edges | **19,592,120** |
| Example | 5,025,497 |
| Ligand | 2,013,247 |
| Scaffold | 645,558 |
| Assay | 857,115 |
| Publication | 66,653 |
| Protein | 5,923 (orphans dropped) |
| ProteinCluster | 8,729 (orphans dropped) |
| ligand_exact | 6,939 |
| ligand_parent_exact | 6,939 |
| **ligand_fingerprint_exact** | **314,683** (NEW) |
| ligand_similar (T 0.85..0.9995) | 448,970 |
| protein_in_cluster | 13,506 |

### Per-dataset contribution

| Corpus | Examples | Actives | Decoys/Inactives | Decoy:Active |
|---|---:|---:|---:|---:|
| LIT-PCBA | 2,651,977 | 7,955 | 2,644,022 | 332:1 |
| DUD-E | 1,434,019 | 22,805 | 1,411,214 | 62:1 |
| BigBind | 582,957 | 489,733 | 93,224 | 0.19:1 (training set) |
| BayesBind | 260,876 | 10,876 | 250,000 | 23:1 |
| DEKOIS | 95,668 | 3,239 | 92,429 | 28:1 |

| Corpus | Unique Ligands | Unique Proteins |
|---|---:|---:|
| DUD-E | 1,200,431 | 102 |
| BigBind | 399,090 | 1,173 |
| LIT-PCBA | 382,742 | 15 |
| DEKOIS | 87,954 | 81 |
| BayesBind | 21,037 | 50 |

Cross-corpus ligand overlap (leakage signal):
- 1,942,516 ligands exclusive to one corpus
- 63,757 in 2 corpora
- 6,686 in 3 corpora
- 274 in 4 corpora
- 14 in all 5 corpora
- **70,731 ligands (3.5%) cross at least two corpora**

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

## Phase 6 — canonical KG wired + cleaned (2026-05-31 00:34)

After the merge audit surfaced 3 issues in the post-Phase-5 canonical KG
(923K orphan Assay/Publication, 5 silently-dropped node types incl.
ChEMBLLigand / BindingDBLigand / ChEMBLTarget, 131 duplicate
ligand_similar edges), this phase reworked `vsleakkg.kg.consolidate` to:

1. Add identity mappings for `Assay` and `Publication` to
   `CORPUS_TO_CANONICAL_NODE_TYPE` so direct BindingDB Assay/Publication
   nodes survive instead of being silently dropped.
2. Wire `_wire_reference_provenance()` to collapse the multi-hop
   reference-DB provenance chains down to direct Example→ref edges:
   - ChEMBL chain: `Example → benchLig → ChEMBLLig → Activity → Doc` →
     emit `example_from_publication`
   - ChEMBL chain: `Example → benchLig → ChEMBLLig → Activity → Assay` →
     emit `example_from_assay` (capped at 5 assays per benchmark Ligand
     to bound memory; ChEMBL has many assays per molecule)
   - BindingDB chain: `Example → benchLig → BDBLig → publication` →
     emit `example_from_publication`
   - BindingDB chain: `Example → benchLig → BDBLig → UniProt` →
     emit `example_has_protein` (canonical UniProt anchor)
3. Universal orphan drop (any node type except `DatasetSource` and
   `DecoyProtocol`) via semi-join — the previous `is_in(Series)` form
   silently kept all orphans because polars 1.x deprecated that
   signature.
4. Targeted dedup on `ligand_similar` only — global
   `unique(subset=[src,dst,edge_type])` on the 38M-edge frame OOMs at
   the 22 GB free RAM available; every other edge type has unique
   (src,dst) by construction, so only the 448K ligand_similar subset
   needs dedup.
5. Time axis explicitly skipped (`TimeBin` / `TrainSet` /
   `example_has_timebin` / `time_overlap` left unused) — user confirmed
   the audit can run without temporal partitioning.

### Final canonical KG

| Metric | Value |
|---|---:|
| Nodes total | **7,951,307** |
| Edges total | **38,888,212** |
| Example | 5,025,497 |
| Ligand | 2,013,247 |
| Scaffold | 645,555 |
| Assay (post-wire, cap=5) | 158,239 |
| Publication (post-wire) | 93,266 |
| ProteinCluster | 8,729 |
| Protein (UniProt) | 6,764 |
| DatasetSource | 7 |
| DecoyProtocol | 3 |

### Wired-axis edge counts

| Edge type | Count | Source |
|---|---:|---|
| example_from_assay | **10,318,968** | wired (ChEMBL Activity, cap=5) |
| example_has_protein | 8,221,977 | direct + wired (BindingDB UniProt) |
| example_from_publication | **5,780,785** | wired (ChEMBL Doc + BindingDB pub) |
| example_from_source | 5,025,497 | direct |
| example_has_ligand | 5,025,493 | direct |
| ligand_scaffold | 1,934,033 | direct |
| source_decoy_protocol | 1,790,563 | direct |
| ligand_similar | 448,829 | similarity D5 |
| ligand_fingerprint_exact | 314,683 | T=1.0 stereo dup relabel |
| protein_in_cluster | 13,506 | MMseqs2 @ 30/50/90% |
| ligand_parent_exact | 6,939 | salt-stripped InChIKey |
| ligand_exact | 6,939 | full InChIKey |

### Per-corpus contribution

| Corpus | Actives | Decoys | Decoy/Active | Unique Ligands | Unique Proteins |
|---|---:|---:|---:|---:|---:|
| LIT-PCBA | 7,955 | 2,644,022 | 332.4 | 382,742 | 2,313 |
| DUD-E | 22,805 | 1,411,214 | 61.9 | 1,200,431 | 2,312 |
| BigBind | 489,733 | 93,224 | 0.19 (training corpus) | 399,090 | 6,365 |
| DEKOIS | 3,239 | 92,429 | 28.5 | 87,954 | 1,227 |
| BayesBind | 10,876 | 250,000 | 23.0 | 21,037 | 2,588 |

Ligand cross-corpus sharing: 1.94M unique to one corpus, 63.8K shared
across 2, 6.7K across 3, 274 across 4, 14 across all 5 corpora.

### Invariants (final_verify.py)

- duplicate node_id: 0
- node_id missing ':': 0
- dangling src: 0
- dangling dst: 0
- duplicate (src, dst, edge_type): 0
- self-loops: 0
- endpoint type mismatch (10 edge-types checked): 0
- orphan nodes (canonical, excluding pinned DatasetSource/DecoyProtocol):
  2 (both DatasetSource — empty corpora loaders)

### Pipeline runtime (Phase 6 consolidate)

- read raw: 0.5s (65.3M edges, 17.7M nodes)
- map + dedup nodes: 0.4s (-8.3M nodes, -45.7M edges via lossy drops)
- cluster edges: 0.1s
- trivial scaffolds: 0.3s (-343 scaffolds with ≤6 heavy atoms)
- hub flag: 6.2s (407 hub nodes)
- dangling prune: 1.5s
- wire reference provenance: 11.2s (+19.3M edges)
- orphan drop: 4.3s (-1.49M orphans)
- ligand_similar dedup: 0.1s (-141 dups)
- write parquet: 5.8s
- **total: 37.9s** for 38.9M-edge canonical KG
