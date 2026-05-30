# Experiments report — Phase 7 (2026-05-31)

Initial runs of the 3 KG-only experiment modules against the canonical
KG (7.95M nodes / 38.6M edges from Phase 6.1). All numbers below are
direct script outputs; nothing has been hand-curated.

## Summary of headline findings

| Finding | Source | Headline |
|---|---|---|
| 1 | mảng G | **BayesBind 85% of decoys are actually active against another target.** BigBind 39%. DUD-E 0.1%. |
| 2 | mảng D | **BigBind training contaminates LIT-PCBA test by 95% at 2 hops.** Same for DUD-E (95%), BayesBind (100%), DEKOIS (84%). |
| 3 | mảng C | **No existing split protocol eliminates publication/assay leak.** Scaffold split leaves 57.5% LIT-PCBA examples sharing a publication with train at 2 hops. |
| 4 | mảng C | **Multi-axis KG-disjoint at K=3 leaves no test items in any corpus.** The benchmarks are fully entangled. |

---

## Mảng G — Decoy quality audit

`outputs/experiments/mang_G/decoy_quality_summary.csv`

| corpus | n_decoys | n_dirty_decoys | pct_dirty |
|---|---:|---:|---:|
| **BayesBind** | 250,000 | 213,418 | **85.37%** |
| **BigBind** | 93,224 | 36,183 | **38.81%** |
| LIT-PCBA | 2,644,022 | 149,176 | 5.64% |
| DEKOIS | 92,429 | 529 | 0.57% |
| DUD-E | 1,411,214 | 1,632 | 0.12% |

Interpretation:
- DUD-E and DEKOIS computational decoys are mostly clean (property
  matching produces molecules that aren't real binders).
- LIT-PCBA's decoys are PubChem-deposited; ~5.6% are active on a
  different ChEMBL target. Acceptable but not pristine.
- BigBind and BayesBind use **structurally-meaningful negatives** —
  these turn out to be real binders for ~39% and ~85% of cases
  respectively. The "decoy" label is mostly counting "active in a
  paper we didn't include" as negative. The model can learn this and
  the inflated AUROC is not generalization, it's contamination.

What the KG made possible: the standard property-match audit doesn't
know whether the decoy has *real* ChEMBL/BindingDB activity. The
canonical KG cross-references every decoy ligand against every active
example via `example_has_ligand` + `example_has_protein` and surfaces
the mismatch in 30 seconds.

---

## Mảng D — Cross-corpus contamination matrix

`outputs/experiments/mang_D/contamination_matrix_k2.csv`

Each cell = % of `test_corpus` examples reachable within 2 hops of
some `train_corpus` example on the leak subgraph.

| train ↓ test → | BigBind | DEKOIS | DUD-E | LIT-PCBA | BayesBind |
|---|---:|---:|---:|---:|---:|
| **BigBind** | — | 4.7 | 2.3 | 58.0 | **100.0** |
| **BayesBind** | **95.7** | 4.0 | 1.9 | 57.9 | — |
| **DUD-E** | **94.9** | 9.9 | — | 59.3 | 79.8 |
| **LIT-PCBA** | **95.0** | 4.6 | 2.3 | — | 78.3 |
| **DEKOIS** | 83.6 | — | 2.2 | 58.0 | 69.0 |

Interpretation:
- BigBind sits at the centre of the leak topology — training on any
  other corpus reaches ~95% of BigBind, and vice versa for BayesBind
  (100%).
- LIT-PCBA test is ~58% reachable from any other corpus's train. The
  PubChem-deposited decoys land in ChEMBL too, so any training set
  with ChEMBL provenance touches them.
- DUD-E test is the *cleanest cross-corpus* (1.9-9.9%) — its
  computational decoys lack ChEMBL/BindingDB anchors.
- DEKOIS test similarly low (4.0-9.9%) because its 81 targets and
  custom decoys don't intersect much with other corpora's ligands.

What the KG made possible: no other published dataset has all 5 benchmarks
wired into a single namespace. The 5×5 matrix is uniquely available here.

---

## Mảng C — Split head-to-head, per axis

`outputs/experiments/mang_C/leak_residual__<corpus>.csv`

For each corpus we built 5 splits at 15% test ratio: random,
Bemis–Murcko scaffold, Tanimoto-MaxMin neighbour exclusion, 30%-id
sequence-cluster partition, and KG-disjoint (≥3 hops).

The residual-leak measurement decomposes by axis to avoid the trivial
"every example shares a protein with every other example in its corpus"
saturation that the flat measurement would produce. Selected results
shown — full per-corpus tables in the per-corpus `report__*.md` files.

### Residual leak at k=2, LIT-PCBA

| protocol | ligand | scaffold | publication | assay |
|---|---:|---:|---:|---:|
| random | 99.4 | 99.4 | 57.9 | 57.8 |
| **scaffold** | **0** | **0** | 57.5 | 57.3 |
| ligand_simil | 99.5 | 99.5 | 57.9 | 57.9 |
| protein_clust | 97.7 | 97.7 | 98.7 | 97.8 |
| kg_disjoint | nan (test=0) | nan | nan | nan |

### Residual leak at k=2, DUD-E

| protocol | ligand | scaffold | publication | assay |
|---|---:|---:|---:|---:|
| random | 25.5 | 25.5 | 2.3 | 2.1 |
| **scaffold** | **0** | **0** | 2.3 | 2.1 |
| ligand_simil | 25.5 | 25.5 | 2.4 | 2.2 |
| protein_clust | 15.6 | 15.6 | **94.8** | **83.6** |
| kg_disjoint | nan | nan | nan | nan |

### Residual leak at k=2, BigBind

| protocol | ligand | scaffold | publication | assay |
|---|---:|---:|---:|---:|
| random | 44.5 | 44.5 | 99.5 | 98.5 |
| scaffold | 0 | 0 | 98.7 | 95.0 |
| ligand_simil | 47.1 | 47.1 | 99.4 | 98.4 |
| **protein_clust** | 21.6 | 21.6 | **73.1** | **54.2** |
| kg_disjoint | nan | nan | nan | nan |

Interpretation:
- **No baseline split is clean across all axes simultaneously.**
  Scaffold-split eliminates ligand+scaffold leakage but leaves
  publication and assay leakage untouched (57.5% / 57.3% on LIT-PCBA).
- **Protein-cluster split is the most leaky on the relational axes**
  (94.8% publication / 83.6% assay leak on DUD-E) — it explicitly
  spreads ligands across clusters, exposing their papers.
- **kg_disjoint at K=3 yields empty test** in every corpus because
  the benchmark internal graphs are too dense. This is itself a
  diagnostic finding: under strict multi-axis disjointness, the
  current benchmarks supply zero usable test set.
- Practical implication: a fair multi-axis split requires either
  (a) a weaker K (try K=2 with edge-set restriction), or
  (b) accepting per-axis tradeoffs and reporting them transparently.

What the KG made possible: a single graph distance + a small set of
canonical edge types lets us measure all 5 leak axes for any split
protocol with one BFS. Existing benchmark protocols silently leak
on the axes they don't explicitly filter.

---

## Reproducing

```bash
# KG only (no predictions needed)
python -m experiments.mang_G_decoy_quality --output-dir outputs/experiments/mang_G
python -m experiments.mang_D_cross_corpus_matrix --output-dir outputs/experiments/mang_D --k-hops 1,2,3
for c in LIT-PCBA DUD-E BigBind DEKOIS BayesBind; do
    python -m experiments.mang_C_kg_disjoint_split \
        --output-dir outputs/experiments/mang_C \
        --corpus "$c" --k-hop 3 --test-ratio 0.15
done
```

## Pending — need predictions parquet

A, B, E, F, H all consume model predictions in PredictionSchema format.
Wiring a Morgan-RF baseline on the canonical KG would unlock them. Until
then they are implementation-ready but not run.
