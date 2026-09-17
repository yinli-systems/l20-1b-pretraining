#!/bin/bash
#SBATCH --job-name=vitg14-frozen-eval
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/vision1b-research/logs/vitg14-frozen-eval-%j.out
#SBATCH --error=/ssd/scxi253/vision1b-research/logs/vitg14-frozen-eval-%j.err

set -euo pipefail

PYTHON=/ssd/scxi253/pretraining2/runtime/venv/bin/python
SOURCE=/ssd/scxi253/vision1b-research/source
DATA_ROOT=/ssd/scxi253/vision1b-research/data/downstream
CHECKPOINT=/ssd/scxi253/vision1b-research/runs/vitg14-real-dino-25k-sinkhorn-fp16-qual-v10-5090-b112/checkpoints/step-000250
CHECKPOINT_SHA256_RECEIPT=/ssd/scxi253/vision1b-research/evaluations/checkpoint-step250-SHA256SUMS
PROTOCOL=$SOURCE/frozen_eval_protocol_v1.json
OUTPUT_ROOT=${OUTPUT_ROOT:-/ssd/scxi253/vision1b-research/evaluations/vitg14-frozen-eval-$SLURM_JOB_ID}
COMPILE_MODE=${COMPILE_MODE:-max-autotune-no-cudagraphs}
BATCH_PER_RANK=${BATCH_PER_RANK:-128}
NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
MAX_TRAIN=${MAX_TRAIN:-0}
MAX_TEST=${MAX_TEST:-0}
VARIANTS=${VARIANTS:-random teacher student}
LOCAL_ROOT=$(mktemp -d "/tmp/vitg14-eval-$SLURM_JOB_ID-XXXXXX")
GPU_KEY=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p' | tr -cs '[:alnum:]' '_')
CACHE_ROOT="/tmp/vision1b-eval-cache-$USER-$GPU_KEY-b$BATCH_PER_RANK"
monitor_pid=

cleanup() {
  if test -n "$monitor_pid"; then kill "$monitor_pid" 2>/dev/null || true; fi
  rm -rf -- "$LOCAL_ROOT"
}
trap cleanup EXIT

mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT" /ssd/scxi253/vision1b-research/logs
cp "$SOURCE/vitg14_preflight.py" "$SOURCE/train_dino_real.py" \
  "$SOURCE/evaluate_frozen_features.py" "$LOCAL_ROOT/"
sha256sum "$LOCAL_ROOT"/*.py "$PROTOCOL" > "$OUTPUT_ROOT/source.sha256"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,clocks.sm \
  --format=csv -l 2 > "$OUTPUT_ROOT/gpu.csv" 2>&1 &
monitor_pid=$!

export PYTHONPATH=/ssd/scxi253/vision1b-research/deps/pillow${PYTHONPATH:+:$PYTHONPATH}
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export OMP_NUM_THREADS=1

read -r -a variant_args <<< "$VARIANTS"
extra_args=()
if test "$MAX_TRAIN" -gt 0; then extra_args+=(--max-train "$MAX_TRAIN"); fi
if test "$MAX_TEST" -gt 0; then extra_args+=(--max-test "$MAX_TEST"); fi

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
  "$LOCAL_ROOT/evaluate_frozen_features.py" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256-receipt "$CHECKPOINT_SHA256_RECEIPT" \
  --protocol "$PROTOCOL" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_ROOT" \
  --batch-per-rank "$BATCH_PER_RANK" \
  --compile-mode "$COMPILE_MODE" \
  --variants "${variant_args[@]}" \
  "${extra_args[@]}"

find "$OUTPUT_ROOT" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/SHA256SUMS"
