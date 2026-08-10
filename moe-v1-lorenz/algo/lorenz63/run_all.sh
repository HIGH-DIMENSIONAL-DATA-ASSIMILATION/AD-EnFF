#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIM="${DIM:-3}"
EPOCHS="${EPOCHS:-100}"
MAX_DIFFICULTY="${MAX_DIFFICULTY:-7}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TBTT="${TBTT:-16}"
SEEDS="${SEEDS:-3}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
RUN_ID="${RUN_ID:-l63_balanced_d0to${MAX_DIFFICULTY}_ep${EPOCHS}}"
OUT="${OUT:-$ROOT/results/$RUN_ID}"
CPU="${CPU:-0}"
SMOKE="${SMOKE:-0}"

if [[ "$SMOKE" == 1 ]]; then
  BATCH_SIZE=8
fi

if (( TBTT < 16 )); then
  echo "TBTT must be at least 16 DA cycles." >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MPLCONFIGDIR="$OUT/.mpl"
mkdir -p "$OUT"/{checkpoints,logs,eval} "$MPLCONFIGDIR"

common=(
  --dim "$DIM" --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE" --window "$TBTT"
  --max-difficulty "$MAX_DIFFICULTY"
  --output-dir "$OUT/checkpoints"
)
eval_flags=()
if [[ "$CPU" == 1 ]]; then common+=(--cpu); eval_flags+=(--cpu); fi
if [[ "$SMOKE" == 1 ]]; then common+=(--smoke); eval_flags+=(--smoke); fi

(
  python -u "$ROOT/mnmef/train_mnmef.py" \
    --run-id "${RUN_ID}_mnmef" "${common[@]}" \
    2>&1 | tee "$OUT/logs/train_mnmef.log"
) &
mnmef_pid=$!

training_status=0
wait "$mnmef_pid" || training_status=$?
if (( training_status != 0 )); then
  echo "MNMEF training failed." >&2
  exit "$training_status"
fi

python -u "$ROOT/ad_enff/train_adaptive.py" \
  --run-id "${RUN_ID}_plain" --no-attention "${common[@]}" \
  2>&1 | tee "$OUT/logs/train_plain.log"

python -u "$ROOT/common/eval_all.py" \
  --run-id "$RUN_ID" --run-dir "$OUT" --dim "$DIM" \
  --difficulty "$MAX_DIFFICULTY" --steps "$EVAL_STEPS" --seeds "$SEEDS" \
  "${eval_flags[@]}" \
  2>&1 | tee "$OUT/logs/eval_all.log"

echo "Results: $OUT"
