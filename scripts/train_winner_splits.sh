#!/usr/bin/env bash
# Train Morgan-RF on each corpus's KG-winner split + random baseline.
# Outputs to data/predictions_v2/.
#
# Winners (from experiments/splits/RESULTS.md):
#   DEKOIS    → kg_kdisjoint strict (K=2 axes=lig,scaf,pub,assay)
#   DUD-E     → kg_kdisjoint strict
#   LIT-PCBA  → kg_kdisjoint structural (K=2 axes=lig,scaf) — strict infeasible
#   BigBind   → kg_kdisjoint structural — strict infeasible
#   BayesBind → kg_kdisjoint structural — strict infeasible

set -eu
cd "$(dirname "$0")/.."

mkdir -p data/predictions_v2 /tmp/train_v2

declare -A WINNER=(
  ["DEKOIS"]="kg_kdisjoint__K2_axesligand,scaffold,publication,assay"
  ["DUD-E"]="kg_kdisjoint__K2_axesligand,scaffold,publication,assay"
  ["LIT-PCBA"]="kg_kdisjoint__K2_axesligand,scaffold"
  ["BigBind"]="kg_kdisjoint__K2_axesligand,scaffold"
  ["BayesBind"]="kg_kdisjoint__K2_axesligand,scaffold"
)

train_one() {
  local corpus="$1"
  local split_name="$2"
  local tag="$3"
  local SPLIT="data/splits/$corpus/${split_name}__seed42.parquet"
  local OUT="data/predictions_v2/morgan_rf__${tag}__${corpus}.parquet"
  if [ ! -f "$SPLIT" ]; then
    echo "MISSING split: $SPLIT" | tee -a /tmp/train_v2/index.log
    return 1
  fi
  echo "[$(date +%H:%M:%S)] start $corpus / $tag" | tee -a /tmp/train_v2/index.log
  python3 -u -m experiments.baselines.morgan_rf \
      --split "$SPLIT" --output "$OUT" --train-cap 15000 \
      > "/tmp/train_v2/${corpus}__${tag}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $corpus / $tag" | tee -a /tmp/train_v2/index.log
}

# Wave A — small corpora in parallel
echo "=== Wave A: train DEKOIS + BayesBind (random + KG winner) ==="
train_one DEKOIS    random__default                     random       &
train_one DEKOIS    "${WINNER[DEKOIS]}"                  kg_winner    &
train_one BayesBind random__default                     random       &
train_one BayesBind "${WINNER[BayesBind]}"              kg_winner    &
wait

# Wave B — medium + large with reduced parallelism
echo "=== Wave B: BigBind + DUD-E (random + KG winner, parallel pairs) ==="
train_one BigBind   random__default                     random       &
train_one BigBind   "${WINNER[BigBind]}"                 kg_winner    &
wait
train_one "DUD-E"   random__default                     random       &
train_one "DUD-E"   "${WINNER[DUD-E]}"                   kg_winner    &
wait

# Wave C — LIT-PCBA alone
echo "=== Wave C: LIT-PCBA (random + KG winner) ==="
train_one "LIT-PCBA" random__default                    random       &
train_one "LIT-PCBA" "${WINNER[LIT-PCBA]}"               kg_winner    &
wait

echo "=== ALL TRAIN DONE ==="
ls -la data/predictions_v2/
