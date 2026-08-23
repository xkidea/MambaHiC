#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
: "${DATA_DIR:?Set DATA_DIR to the GM12878 preprocessed .pkl directory}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLCONFIGDIR="$ROOT/.matplotlib"
export XDG_CACHE_HOME="$ROOT/.cache"

mkdir -p "$ROOT/logs" "$ROOT/results"

MODES_TO_RUN="${MODES:-dual dnase h3k27ac h3k4me3}"
for mode in $MODES_TO_RUN; do
  echo "[$(date -Is)] starting $mode on six GPUs"
  pids=()
  for gpu in 0 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT/code/alpha_sweep_worker.py" \
      --mode "$mode" --shard-id "$gpu" --shard-count 6 --gpu-id 0 \
      --data-dir "$DATA_DIR" \
      >"$ROOT/logs/${mode}_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [ "$status" -ne 0 ]; then
    echo "A worker failed for $mode; see $ROOT/logs" >&2
    exit "$status"
  fi
  "$PYTHON" "$ROOT/code/aggregate_results.py" --mode "$mode"
  echo "[$(date -Is)] completed $mode"
done

if [ "$MODES_TO_RUN" = "dual dnase h3k27ac h3k4me3" ]; then
  "$PYTHON" "$ROOT/code/aggregate_results.py" --mode all
  echo "[$(date -Is)] all modes completed"
fi
