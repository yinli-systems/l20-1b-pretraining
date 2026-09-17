#!/bin/bash
#SBATCH --job-name=vitg14-real-dino
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/vision1b-research/logs/vitg14-real-dino-%j.out
#SBATCH --error=/ssd/scxi253/vision1b-research/logs/vitg14-real-dino-%j.err

set -euo pipefail

PYTHON=/ssd/scxi253/pretraining2/runtime/venv/bin/python
SOURCE=/ssd/scxi253/vision1b-research/source
DATA=${DATA:-/ssd/scxi253/vision1b-research/data/openimages-real-pilot-v1}
PROTOCOL=${PROTOCOL:-$SOURCE/real_data_pilot_protocol.json}
BATCH_PER_RANK=${BATCH_PER_RANK:-64}
WORKERS_PER_RANK=${WORKERS_PER_RANK:-2}
STEPS=${STEPS:-250}
STOP_AFTER=${STOP_AFTER:-0}
SAVE_EVERY=${SAVE_EVERY:-25}
COMPILE_MODE=${COMPILE_MODE:-max-autotune-no-cudagraphs}
PEAK_BF16_TFLOPS_PER_GPU=${PEAK_BF16_TFLOPS_PER_GPU:-209.5}
PRECISION=${PRECISION:-fp16}
LOSS_SCALE=${LOSS_SCALE:-1024}
RESUME=${RESUME:-}
ROOT=${OUTPUT_ROOT:-/ssd/scxi253/vision1b-research/runs/vitg14-real-dino-4x5090-$SLURM_JOB_ID}
LOCAL_ROOT=$(mktemp -d "/tmp/vitg14-real-$SLURM_JOB_ID-XXXXXX")
GPU_KEY=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p' | tr -cs '[:alnum:]' '_')
CACHE_ROOT="/tmp/vision1b-cache-$USER-$GPU_KEY-b$BATCH_PER_RANK-$PRECISION"
monitor_pid=

cleanup() {
  if test -n "$monitor_pid"; then kill "$monitor_pid" 2>/dev/null || true; fi
  rm -rf -- "$LOCAL_ROOT"
}
trap cleanup EXIT

mkdir -p "$ROOT" "$CACHE_ROOT" /ssd/scxi253/vision1b-research/logs
cp "$SOURCE/vitg14_preflight.py" "$SOURCE/train_dino_real.py" "$LOCAL_ROOT/"
sha256sum "$LOCAL_ROOT/vitg14_preflight.py" "$LOCAL_ROOT/train_dino_real.py" > "$ROOT/source.sha256"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,clocks.sm --format=csv -l 2 > "$ROOT/gpu.csv" 2>&1 &
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
unset CUDA_DEVICE_MAX_CONNECTIONS || true

resume_args=()
if test -n "$RESUME"; then resume_args=(--resume "$RESUME"); fi

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$LOCAL_ROOT/train_dino_real.py" \
  --manifest "$DATA/manifest.jsonl" \
  --protocol "$PROTOCOL" \
  --acquisition-receipt "$DATA/acquisition-receipt.json" \
  --output-dir "$ROOT" \
  --steps "$STEPS" \
  --stop-after "$STOP_AFTER" \
  --batch-per-rank "$BATCH_PER_RANK" \
  --workers-per-rank "$WORKERS_PER_RANK" \
  --save-every "$SAVE_EVERY" \
  --warmup-steps 10 \
  --compile-mode "$COMPILE_MODE" \
  --precision "$PRECISION" \
  --loss-scale "$LOSS_SCALE" \
  --peak-dense-tflops-per-gpu "$PEAK_BF16_TFLOPS_PER_GPU" \
  "${resume_args[@]}"

# DCP shards are multi-gigabyte and carry their own distributed metadata.
# Hash small run evidence here so the GPU allocation is not held by tail I/O.
find "$ROOT" -path "$ROOT/checkpoints" -prune -o -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > "$ROOT/SHA256SUMS"
