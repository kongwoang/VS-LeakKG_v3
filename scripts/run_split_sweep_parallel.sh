#!/usr/bin/env bash
# Parallel split-sweep on VUW.
#
# Strategy:
#   Wave 1 (parallel): DEKOIS + BayesBind + BigBind  (each fits in ~20GB)
#   Wave 2 (parallel): DUD-E alone (~25GB)
#   Wave 3 (sequential): LIT-PCBA (~30GB, heaviest)
#
# Each corpus writes to data/splits/audit_summary__<CORPUS>.csv and to
# data/splits/<CORPUS>/<protocol>__<params>__seed<n>.parquet. After all
# corpora finish, merge_audit.py consolidates per-corpus files.

set -eu

cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-data/splits}"
SEEDS="${SEEDS:-42,43,44,45,46}"
LOG_DIR="${LOG_DIR:-/tmp/split_sweep}"
mkdir -p "$LOG_DIR" "$DATA_DIR"

run_corpus() {
  local corpus="$1"
  echo "[$(date +%H:%M:%S)] starting $corpus" >> "$LOG_DIR/index.log"
  python3 -u -m experiments.splits.runner \
      --corpora "$corpus" \
      --seeds "$SEEDS" \
      --data-dir "$DATA_DIR" \
      > "$LOG_DIR/$corpus.log" 2>&1
  echo "[$(date +%H:%M:%S)] finished $corpus" >> "$LOG_DIR/index.log"
}

echo "=== Wave 1: DEKOIS + BayesBind + BigBind (parallel) ==="
run_corpus DEKOIS    &
run_corpus BayesBind &
run_corpus BigBind   &
wait

echo "=== Wave 2: DUD-E (alone) ==="
run_corpus "DUD-E"

echo "=== Wave 3: LIT-PCBA (alone) ==="
run_corpus "LIT-PCBA"

echo "=== Merging audit files ==="
python3 -u -m experiments.splits.merge_audit --data-dir "$DATA_DIR"

echo "=== ALL CORPORA DONE ==="
