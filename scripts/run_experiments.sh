#!/usr/bin/env bash
# Run all KG-leak experiments on a built set of predictions.
# Expects:
#   - outputs/predictions/{morgan_rf,cnn}__random__{DEKOIS,BayesBind,BigBind,DUD-E,LIT-PCBA}.parquet
# Writes:
#   - outputs/experiments/mang_{A,B,E,F}__<corpus>/...     (Morgan-RF audits)
#   - outputs/experiments/mang_{A,F}_cnn__<corpus>/...     (C-NN audits)
#   - outputs/experiments/baseline_compare/...

set -eu

CORPORA="DEKOIS BayesBind BigBind DUD-E LIT-PCBA"
PRED_DIR="outputs/predictions"
EXP_DIR="outputs/experiments"

# Morgan-RF experiments
for c in $CORPORA; do
  PRED="$PRED_DIR/morgan_rf__random__$c.parquet"
  [ -f "$PRED" ] || { echo "skip morgan_rf $c (no predictions)"; continue; }

  echo "=== mang_A morgan_rf $c ==="
  python3 -u -m experiments.mang_A_publication_audit \
      --predictions "$PRED" --output-dir "$EXP_DIR/mang_A__$c" 2>&1 | tail -3

  echo "=== mang_E morgan_rf $c ==="
  python3 -u -m experiments.mang_E_horizon_curve \
      --predictions "$PRED" --output-dir "$EXP_DIR/mang_E__$c" --max-hop 4 2>&1 | tail -3

  echo "=== mang_F morgan_rf $c ==="
  python3 -u -m experiments.mang_F_hub_audit \
      --predictions "$PRED" --output-dir "$EXP_DIR/mang_F__$c" --hub-distance 2 2>&1 | tail -3

  echo "=== mang_B morgan_rf $c ==="
  python3 -u -m experiments.mang_B_path_atlas \
      --predictions "$PRED" --output-dir "$EXP_DIR/mang_B__$c" --max-hop 3 2>&1 | tail -3
done

# C-NN experiments — A/F only (B and E are less meaningful on a label-prop baseline)
for c in $CORPORA; do
  PRED="$PRED_DIR/cnn__random__$c.parquet"
  [ -f "$PRED" ] || { echo "skip cnn $c (no predictions)"; continue; }

  echo "=== mang_A cnn $c ==="
  python3 -u -m experiments.mang_A_publication_audit \
      --predictions "$PRED" --output-dir "$EXP_DIR/mang_A_cnn__$c" 2>&1 | tail -3

  echo "=== mang_F cnn $c ==="
  python3 -u -m experiments.mang_F_hub_audit \
      --predictions "$PRED" --output-dir "$EXP_DIR/mang_F_cnn__$c" --hub-distance 2 2>&1 | tail -3
done

echo "=== compare_baselines ==="
python3 -u -m experiments.compare_baselines 2>&1 | tail -20

echo "=== ALL EXPERIMENTS DONE ==="
