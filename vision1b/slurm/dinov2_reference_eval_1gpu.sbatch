#!/bin/bash
#SBATCH --job-name=dinov2-ref-eval
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/ssd/scxi253/vision1b-research/logs/dinov2-ref-eval-%j.out
#SBATCH --error=/ssd/scxi253/vision1b-research/logs/dinov2-ref-eval-%j.err

set -euo pipefail

PYTHON=/ssd/scxi253/pretraining2/runtime/venv/bin/python
SOURCE=/ssd/scxi253/vision1b-research/source
DATA_ROOT=/ssd/scxi253/vision1b-research/data/downstream
DINOV2_SOURCE=/ssd/scxi253/vision1b-research/deps/dinov2-7764ea0f
WEIGHTS_ROOT=/ssd/scxi253/vision1b-research/deps/dinov2-weights
MODEL_ID=${MODEL_ID:?MODEL_ID is required}
WEIGHTS_FILE=${WEIGHTS_FILE:?WEIGHTS_FILE is required}
WEIGHTS_SHA256=${WEIGHTS_SHA256:?WEIGHTS_SHA256 is required}
BATCH_SIZE=${BATCH_SIZE:?BATCH_SIZE is required}
OUTPUT_ROOT=${OUTPUT_ROOT:-/ssd/scxi253/vision1b-research/evaluations/$MODEL_ID-$SLURM_JOB_ID}
MAX_TRAIN=${MAX_TRAIN:-0}
MAX_TEST=${MAX_TEST:-0}
LOCAL_ROOT=$(mktemp -d "/tmp/dinov2-ref-$SLURM_JOB_ID-XXXXXX")
monitor_pid=

cleanup() {
  if test -n "$monitor_pid"; then kill "$monitor_pid" 2>/dev/null || true; fi
  rm -rf -- "$LOCAL_ROOT"
}
trap cleanup EXIT

mkdir -p "$OUTPUT_ROOT" /ssd/scxi253/vision1b-research/logs
cp "$SOURCE/evaluate_frozen_features.py" "$SOURCE/evaluate_reference_backbone.py" "$LOCAL_ROOT/"
sha256sum "$LOCAL_ROOT"/*.py "$SOURCE/frozen_eval_protocol_v1.json" \
  "$SOURCE/reference_baseline_protocol_v1.json" > "$OUTPUT_ROOT/source.sha256"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,clocks.sm \
  --format=csv -l 2 > "$OUTPUT_ROOT/gpu.csv" 2>&1 &
monitor_pid=$!

export PYTHONPATH="$LOCAL_ROOT:/ssd/scxi253/vision1b-research/deps/pillow:$DINOV2_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

extra_args=()
if test "$MAX_TRAIN" -gt 0; then extra_args+=(--max-train "$MAX_TRAIN"); fi
if test "$MAX_TEST" -gt 0; then extra_args+=(--max-test "$MAX_TEST"); fi

"$PYTHON" "$LOCAL_ROOT/evaluate_reference_backbone.py" \
  --model-id "$MODEL_ID" \
  --weights "$WEIGHTS_ROOT/$WEIGHTS_FILE" \
  --expected-weights-sha256 "$WEIGHTS_SHA256" \
  --dinov2-source "$DINOV2_SOURCE" \
  --reference-protocol "$SOURCE/reference_baseline_protocol_v1.json" \
  --base-protocol "$SOURCE/frozen_eval_protocol_v1.json" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_ROOT" \
  --batch-size "$BATCH_SIZE" \
  "${extra_args[@]}"

find "$OUTPUT_ROOT" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/SHA256SUMS"
