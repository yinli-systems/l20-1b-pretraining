#!/bin/bash
#SBATCH --job-name=resnet50-ref-eval
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/vision1b-research/logs/resnet50-ref-eval-%j.out
#SBATCH --error=/ssd/scxi253/vision1b-research/logs/resnet50-ref-eval-%j.err

set -euo pipefail

PYTHON=/ssd/scxi253/pretraining2/runtime/venv/bin/python
SOURCE=/ssd/scxi253/vision1b-research/source
DATA_ROOT=/ssd/scxi253/vision1b-research/data/downstream
REFERENCE_PYTHON=/ssd/scxi253/vision1b-research/deps/reference-python
PILLOW=/ssd/scxi253/vision1b-research/deps/pillow
WEIGHTS=/ssd/scxi253/vision1b-research/deps/torchvision-weights/resnet50-11ad3fa6.pth
WHEEL=/ssd/scxi253/vision1b-research/deps/wheels/torchvision-0.26.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl
OUTPUT_ROOT=${OUTPUT_ROOT:-/ssd/scxi253/vision1b-research/evaluations/resnet50-reference-$SLURM_JOB_ID}
BATCH_SIZE=${BATCH_SIZE:-512}
MAX_TRAIN=${MAX_TRAIN:-0}
MAX_TEST=${MAX_TEST:-0}
LOCAL_ROOT=$(mktemp -d "/tmp/resnet50-ref-$SLURM_JOB_ID-XXXXXX")
monitor_pid=

cleanup() {
  if test -n "$monitor_pid"; then kill "$monitor_pid" 2>/dev/null || true; fi
  rm -rf -- "$LOCAL_ROOT"
}
trap cleanup EXIT

mkdir -p "$OUTPUT_ROOT" /ssd/scxi253/vision1b-research/logs
cp "$SOURCE/evaluate_frozen_features.py" "$SOURCE/evaluate_reference_backbone.py" \
  "$SOURCE/evaluate_resnet_reference.py" "$LOCAL_ROOT/"
sha256sum "$LOCAL_ROOT"/*.py "$SOURCE/frozen_eval_protocol_v1.json" \
  "$SOURCE/reference_supervised_protocol_v1.json" > "$OUTPUT_ROOT/source.sha256"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,clocks.sm \
  --format=csv -l 2 > "$OUTPUT_ROOT/gpu.csv" 2>&1 &
monitor_pid=$!

export PYTHONPATH="$LOCAL_ROOT:$REFERENCE_PYTHON:$PILLOW${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

extra_args=()
if test "$MAX_TRAIN" -gt 0; then extra_args+=(--max-train "$MAX_TRAIN"); fi
if test "$MAX_TEST" -gt 0; then extra_args+=(--max-test "$MAX_TEST"); fi

"$PYTHON" "$LOCAL_ROOT/evaluate_resnet_reference.py" \
  --weights "$WEIGHTS" \
  --expected-weights-sha256 11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca \
  --torchvision-wheel "$WHEEL" \
  --expected-wheel-sha256 ccf26b4b659cfce6f2208cb8326071d51c70219a34856dfdf468d1e19af52c0d \
  --reference-protocol "$SOURCE/reference_supervised_protocol_v1.json" \
  --base-protocol "$SOURCE/frozen_eval_protocol_v1.json" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_ROOT" \
  --batch-size "$BATCH_SIZE" \
  "${extra_args[@]}"

find "$OUTPUT_ROOT" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/SHA256SUMS"
