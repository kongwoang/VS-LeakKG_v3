# Split protocol head-to-head — 14-config benchmark

Single-seed (seed=42) audit of 14 split protocols × 5 corpora on the
canonical KG. Each protocol produces (train, test) for one corpus
using only that corpus's examples; KG is used only for measuring
residual per-axis leak.

## Stack

**Nhóm 1 — off-the-shelf baselines (8):**
1. `random` — stratified random by label
2. `random_per_target` — stratified random within each protein target
3. `scaffold` — Bemis-Murcko scaffold partition
4. `scaffold_generic` — generic Murcko framework (RDKit MakeScaffoldGeneric)
5. `tanimoto_maxmin` — ligand-similarity neighbour exclusion (T=0.4)
6. `protein_cluster_30` — MMseqs2 30% sequence-identity cluster
7. `protein_cluster_50` — same at 50%
8. `protein_cluster_90` — same at 90%

**Nhóm 3 — KG protocols (3 algorithms × 2 axis modes = 6):**
9.  `kg_kdisjoint  structural` — K=2 disjoint on (ligand, scaffold)
10. `kg_kdisjoint  strict`     — K=2 disjoint on (lig, scaf, publication, assay)
11. `kg_maxmin    structural` — T=3 MaxMin on (ligand, scaffold)
12. `kg_maxmin    strict`     — T=3 MaxMin on (lig, scaf, publication, assay)
13. `kg_axis_budget structural` — K=2 per-axis, pub/assay unconstrained
14. `kg_axis_budget strict`     — K=2 per-axis, all axes budgeted

Direct `example_has_protein` is intentionally excluded from KG-disjoint /
MaxMin walks — it trivially saturates K=2 in any single-corpus split
(every example shares its target with some train example).

DataSAIL / PLINDER / AVE (Nhóm 2 paper baselines) are excluded — the
current ports are simplified fallbacks and would misrepresent prior art.

## Headline result (per corpus, seed=42)

Residual leak (% of test reachable from train within k=2 hops on each
axis subgraph) and feasibility (n_test ≥ 500, ≥10 actives, ≥10 decoys).

### DEKOIS (95,668 examples, 3.4% active)

| protocol | n_test | n_active | feasible | lig % | pub % | assay % |
|---|---:|---:|:---:|---:|---:|---:|
| random | 14,350 | 486 | ✓ | 12.8 | 4.0 | 3.6 |
| random_per_target | 14,278 | 455 | ✓ | 12.4 | 3.6 | 3.2 |
| scaffold | 13,772 | 495 | ✓ | **0** | 3.7 | 3.2 |
| scaffold_generic | 14,177 | 375 | ✓ | **0** | 3.2 | 2.8 |
| tanimoto_maxmin | 14,201 | 481 | ✓ | 12.8 | 4.0 | 3.6 |
| protein_cluster_30 | 132 | 74 | ✗ | 14 | 78 | 61 |
| protein_cluster_50 | 145 | 75 | ✗ | 13 | 75 | 61 |
| protein_cluster_90 | 186 | 107 | ✗ | 14 | 83 | 65 |
| **kg_kdisjoint structural** | 12,509 | 435 | ✓ | **0** | 3.6 | 3.2 |
| **kg_kdisjoint strict** | 12,049 | 225 | ✓ | **0** | **0** | **0** ✅ |
| kg_maxmin structural | 12,509 | 435 | ✓ | **0** | 3.6 | 3.2 |
| kg_maxmin strict | 12,049 | 225 | ✓ | **0** | **0** | **0** ✅ |
| kg_axis_budget structural | 6,585 | 230 | ✓ | 5.6 | 3.8 | 3.4 |
| kg_axis_budget strict | 6,585 | 220 | ✓ | 6.2 | 3.9 | 3.5 |

→ **kg_kdisjoint strict / kg_maxmin strict: 0% on all 4 axes, 12K feasible test.**

### DUD-E (1,434,019 examples, 1.6% active)

| protocol | n_test | n_active | feasible | lig % | pub % | assay % |
|---|---:|---:|:---:|---:|---:|---:|
| random | 215,103 | 3,421 | ✓ | 25.5 | 2.3 | 2.1 |
| random_per_target | 214,933 | 3,330 | ✓ | 25.5 | 2.2 | 2.0 |
| scaffold | 203,014 | 3,370 | ✓ | **0** | 2.2 | 2.1 |
| scaffold_generic | 200,015 | 3,264 | ✓ | **0** | 2.2 | 2.0 |
| tanimoto_maxmin | 196,160 | 3,200 | ✓ | 25.5 | 2.4 | 2.2 |
| protein_cluster_30 | 1,190 | 986 | ✓ | 11 | **92** | **80** |
| protein_cluster_50 | 1,398 | 1,169 | ✓ | 18 | **95** | **86** |
| protein_cluster_90 | 1,732 | 1,530 | ✓ | 18 | **95** | **84** |
| **kg_kdisjoint structural** | 160,177 | 2,816 | ✓ | **0** | 2.5 | 2.3 |
| **kg_kdisjoint strict** | 156,117 | 205 | ✓ | **0** | **0** | **0** ✅ |
| kg_maxmin structural | 160,177 | 2,816 | ✓ | **0** | 2.5 | 2.3 |
| kg_maxmin strict | 156,117 | 205 | ✓ | **0** | **0** | **0** ✅ |
| kg_axis_budget structural | 84,361 | 1,444 | ✓ | 7.0 | 2.5 | 2.3 |
| kg_axis_budget strict | 84,361 | 1,464 | ✓ | 7.1 | 2.5 | 2.3 |

→ **kg_kdisjoint strict / kg_maxmin strict: 0% on all 4 axes, 156K feasible test.**

### LIT-PCBA (2,651,977 examples, 0.30% active)

| protocol | n_test | n_active | feasible | lig % | pub % | assay % |
|---|---:|---:|:---:|---:|---:|---:|
| random | 397,796 | 1,193 | ✓ | 99.4 | **57.9** | **57.8** |
| random_per_target | 397,670 | 1,159 | ✓ | 99.4 | 57.9 | 57.9 |
| scaffold | 388,470 | 1,156 | ✓ | **0** | 58.5 | 58.3 |
| scaffold_generic | 363,699 | 1,071 | ✓ | **0** | 59.1 | 58.9 |
| tanimoto_maxmin | 390,943 | 1,184 | ✓ | 99.5 | 57.9 | 57.9 |
| protein_cluster_30 | 17,879 | 116 | ✓ | 98.6 | **97.4** | **97.1** |
| protein_cluster_50 | 15,485 | 81 | ✓ | 98.3 | 97.6 | 97.2 |
| protein_cluster_90 | 9,685 | 53 | ✓ | 98.8 | 97.7 | 97.3 |
| **kg_kdisjoint structural** | 2,302 | 10 | ✓ | **0** | 50.0 | 49.6 |
| kg_kdisjoint strict | 1,152 | 6 | ✗ | 0 | 0 | 0 |
| **kg_maxmin structural** | 2,302 | 10 | ✓ | **0** | 50.0 | 49.6 |
| kg_maxmin strict | 1,152 | 6 | ✗ | 0 | 0 | 0 |
| kg_axis_budget structural | 1,705 | 4 | ✗ | 43 | 52 | 52 |
| kg_axis_budget strict | 851 | 3 | ✗ | 39 | 5.5 | 5.5 |

→ **kg_kdisjoint structural / kg_maxmin structural: 0% chemistry leak, 2.3K feasible test.** Strict variants drop below 10 actives.

### BigBind (582,957 examples, 84% active)

| protocol | n_test | n_active | feasible | lig % | pub % | assay % |
|---|---:|---:|:---:|---:|---:|---:|
| random | 87,444 | 73,460 | ✓ | 44.5 | **99.5** | **98.5** |
| random_per_target | 87,091 | 73,278 | ✓ | 44.6 | 99.5 | 98.6 |
| scaffold | 87,015 | 73,719 | ✓ | **0** | 98.8 | 94.4 |
| scaffold_generic | 94,816 | 78,696 | ✓ | **0** | 97.4 | 90.9 |
| tanimoto_maxmin | 78,296 | 64,937 | ✓ | 47.1 | 99.4 | 98.4 |
| protein_cluster_30 | 90,345 | 80,539 | ✓ | 33.4 | 81.6 | 66.8 |
| protein_cluster_50 | 82,897 | 73,502 | ✓ | 33.1 | 83.6 | 66.0 |
| protein_cluster_90 | 96,046 | 81,626 | ✓ | 42.9 | 90.5 | 78.6 |
| **kg_kdisjoint structural** | 48,505 | 41,060 | ✓ | **0** | 99.4 | 98.1 |
| kg_kdisjoint strict | 273 | 129 | ✗ | 0 | 0 | 0 |
| kg_maxmin structural | 48,505 | 41,060 | ✓ | **0** | 99.4 | 98.1 |
| kg_maxmin strict | 273 | 129 | ✗ | 0 | 0 | 0 |
| kg_axis_budget structural | 25,575 | 21,669 | ✓ | 8.3 | 99.5 | 98.2 |
| kg_axis_budget strict | 223 | 138 | ✗ | 5 | 38 | 34 |

→ **kg_kdisjoint structural / kg_maxmin structural: 0% chemistry, 48K feasible test.** Strict variants too restrictive → test < 300.

### BayesBind (260,876 examples, 4.2% active)

| protocol | n_test | n_active | feasible | lig % | pub % | assay % |
|---|---:|---:|:---:|---:|---:|---:|
| random | 39,131 | 1,631 | ✓ | 96.6 | **98.7** | **98.3** |
| random_per_target | 38,971 | 1,524 | ✓ | 96.7 | 98.8 | 98.5 |
| scaffold | 37,512 | 1,584 | ✓ | **0** | 90.6 | 82.3 |
| scaffold_generic | 40,779 | 1,720 | ✓ | **0** | 87.0 | 76.6 |
| tanimoto_maxmin | 39,094 | 1,622 | ✓ | 96.6 | 98.7 | 98.3 |
| protein_cluster_30 | 13,795 | 3,764 | ✓ | 81 | 97.7 | 96.9 |
| protein_cluster_50 | 19,709 | 1,415 | ✓ | 94.5 | 99.4 | 98.3 |
| protein_cluster_90 | 13,394 | 728 | ✓ | 95.7 | 99.1 | 98.1 |
| **kg_kdisjoint structural** | 1,342 | 999 | ✓ | **0** | 94.8 | 90.2 |
| kg_kdisjoint strict | 68 | 51 | ✗ | 0 | 0 | 0 |
| kg_maxmin structural | 1,342 | 999 | ✓ | **0** | 94.8 | 90.2 |
| kg_maxmin strict | 68 | 51 | ✗ | 0 | 0 | 0 |
| kg_axis_budget structural | 753 | 498 | ✓ | 13.5 | 95.4 | 90.7 |
| kg_axis_budget strict | 40 | 32 | ✗ | 5 | 15 | 8 |

→ **kg_kdisjoint structural / kg_maxmin structural: 0% chemistry, 1.3K feasible test.** Strict variants infeasible — BayesBind too entangled at the paper/assay level.

## Cross-corpus pattern

| Corpus | Strict feasible? | Best feasible protocol | Best leak profile |
|---|:---:|---|---|
| DEKOIS | ✅ | `kg_kdisjoint strict` | 0% on all 4 axes, 12K test |
| DUD-E | ✅ | `kg_kdisjoint strict` | 0% on all 4 axes, 156K test |
| LIT-PCBA | ❌ | `kg_kdisjoint structural` | 0% chemistry, ~50% pub/assay |
| BigBind | ❌ | `kg_kdisjoint structural` | 0% chemistry, 99% pub |
| BayesBind | ❌ | `kg_kdisjoint structural` | 0% chemistry, 95% pub |

**Two-tier picture:**
1. **DEKOIS, DUD-E** — KG protocols achieve 0% leak on ALL four axes with
   a feasible (12K–156K) test set. These benchmarks are clean-able.
2. **LIT-PCBA, BigBind, BayesBind** — KG protocols achieve 0% chemistry
   leak, but achieving 0% on publication/assay shrinks the test below
   feasibility (10 active threshold). These benchmarks are
   **fundamentally entangled at the relational source level**.

## Why baselines fail

- `random` / `random_per_target` / `tanimoto_maxmin` — no axis filtering
- `scaffold` / `scaffold_generic` — filter ligand+scaffold but leave
  publication and assay leak untouched (3% on DEKOIS, 99% on BigBind)
- `protein_cluster_*` — splits by target cluster, which actually
  *amplifies* publication leak (different targets, but same paper
  pools): 78–97% pub leak across all corpora
- KG protocols are the only ones that filter the relational axes
  directly

## Files

- `data/splits/<corpus>/<protocol>__<params>__seed42.parquet` — 70 split
  parquets (14 protocols × 5 corpora, seed=42 only)
- `data/splits/audit_summary.csv` — per-row leak metrics for all
  protocols across all corpora (331 rows incl. legacy entries)

## Next step

Train Morgan-RF + C-NN on the WINNER split per corpus + random baseline
to confirm the "harder for models" part of the claim (AUROC should drop
proportional to leak removed).
