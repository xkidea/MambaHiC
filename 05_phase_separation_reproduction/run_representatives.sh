#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
: "${DATA_DIR:?Set DATA_DIR to the GM12878 preprocessed .pkl directory}"
MODES=(dual dnase h3k27ac h3k4me3)

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLCONFIGDIR="$ROOT/.matplotlib"
export XDG_CACHE_HOME="$ROOT/.cache"

mkdir -p "$ROOT/logs" "$ROOT/results"

pids=()
for index in "${!MODES[@]}"; do
  mode="${MODES[$index]}"
  CUDA_VISIBLE_DEVICES="$index" "$PYTHON" "$ROOT/code/alpha_sweep_worker.py" \
    --mode "$mode" --shard-id 0 --shard-count 6 --gpu-id 0 --representatives-only \
    --data-dir "$DATA_DIR" \
    >"$ROOT/logs/${mode}_representatives.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [ "$status" -ne 0 ]; then
  echo "A representative-map worker failed; see $ROOT/logs" >&2
  exit "$status"
fi

for mode in "${MODES[@]}"; do
  "$PYTHON" "$ROOT/code/aggregate_results.py" --mode "$mode"
done
