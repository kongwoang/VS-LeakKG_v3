# experiments/

Multi-axis leak audit + split experiments built on the canonical KG
(`outputs/kg/canonical_{nodes,edges}.parquet`).

## Layout

```
experiments/
├── README.md
├── common/                       # shared utilities (KG loader, BFS, stats, predictions)
├── mang_A_publication_audit.py   # paper / assay relational leak audit
├── mang_B_path_atlas.py          # path-typed mispredict atlas
├── mang_C_kg_disjoint_split.py   # KG-distance split + head-to-head vs baselines
├── mang_D_cross_corpus_matrix.py # 5×5 corpus contamination matrix
├── mang_E_horizon_curve.py       # AUROC vs KG-distance horizon
├── mang_F_hub_audit.py           # hub-driven leak audit
├── mang_G_decoy_quality.py       # decoy-as-active-elsewhere audit
└── mang_H_retrospective.py       # decompose published AUROCs into clean/inflated
```

## Mapping to the 8 directions

| ID | Mảng | Question | KG-only? |
|---|---|---|---|
| A | Publication / Assay audit | Is the test contaminated through shared papers/assays? | Yes |
| B | Path-typed atlas | When the model fails, which KG path explains it? | Yes |
| C | KG-disjoint split | Can a graph-distance split remove leaks baseline splits miss? | Yes |
| D | Cross-corpus matrix | How much of corpus X test leaks via corpus Y train? | Yes |
| E | Horizon curve | What's AUROC as a function of KG-distance to nearest train? | Yes |
| F | Hub-leak audit | Do hub nodes act as leak conduits? | Yes |
| G | Decoy quality | Are "decoys" really decoys, or active elsewhere? | Yes |
| H | Retrospective decomposition | What fraction of a published AUROC is leak? | Yes |

## Input contract

Every experiment that needs model outputs accepts a single parquet:

```
predictions/<model>__<corpus>.parquet
columns: example_id (str), score (float), label (int 0|1), fold (str "train"|"test")
```

This decouples KG audit from any specific model. Plug in Morgan-RF, DeepDTA,
KG-GNN, or a published model's outputs identically.

## Output contract

Each `mang_X_*.py` writes under `outputs/experiments/<mang>/`:
- `*.parquet` — per-example tags / scores
- `*.csv` — aggregate tables
- `*.md` — auto-generated short report

## Running

Each script is a self-contained CLI. Common entry point:

```bash
python -m experiments.mang_A_publication_audit \
    --predictions predictions/morgan_rf__litpcba.parquet \
    --output-dir outputs/experiments/mang_A
```

Some experiments (D, E, F, G) don't need predictions and only need the KG.
