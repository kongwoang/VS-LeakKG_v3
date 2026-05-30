"""Baseline models for the experiment harness.

Each baseline consumes a split parquet (columns: node_id, fold) from
`experiments.mang_C_kg_disjoint_split` or any compatible source, and
emits a predictions parquet conforming to `PredictionSchema` (columns:
example_id, score, label, fold, model).

Available baselines:
  - morgan_rf : Morgan-FP + Random Forest (ported from v2)
  - morgan_lr : Morgan-FP + Logistic Regression
  - cnn       : KG-proximity nearest-neighbour (predicts test label from
                mean label of train within K hops on the leak subgraph;
                exposes how much "free signal" is in the KG structure).
"""
