#!/bin/bash
#SBATCH --job-name=vision1b-cont1000-r1
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/ssd/scxi253/vision1b-research/logs/vision1b-cont1000-r1-%j.out
#SBATCH --error=/ssd/scxi253/vision1b-research/logs/vision1b-cont1000-r1-%j.err

set -euo pipefail

PYTHON=${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}
SOURCE=${SOURCE:?set SOURCE to an immutable continuation source}
DATA=${DATA:-/ssd/scxi253/vision1b-research/data/openimages-real-25k-v1}
PROTOCOL=${PROTOCOL:-$SOURCE/real_data_25k_protocol.json}
CONTINUATION=${CONTINUATION:-$SOURCE/continuation_25k_stage2_protocol_v1.json}
RESUME=/ssd/scxi253/vision1b-research/runs/vitg14-real-dino-25k-sinkhorn-fp16-qual-v10-5090-b112/checkpoints/step-000250
RESUME_SHA256S="$RESUME/../step-000250.SHA256SUMS"
RESUME_SHA256S_SHA256=c1e71d6c98091806ea35d1edba74fad256f828c7ff306d0d58722d21dcaaa0f1
ROOT=${OUTPUT_ROOT:-/ssd/scxi253/vision1b-research/runs/vision1b-25k-cont1000-r1-$SLURM_JOB_ID}
LOCAL_ROOT=$(mktemp -d "/tmp/vision1b-cont-r1-$SLURM_JOB_ID-XXXXXX")
CACHE_ROOT=$(mktemp -d "/tmp/vision1b-cont-cache-$USER-$SLURM_JOB_ID-XXXXXX")
monitor_pid=

cleanup() {
  if test -n "$monitor_pid"; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  chmod -R u+w "$LOCAL_ROOT" "$CACHE_ROOT" 2>/dev/null || true
  rm -rf -- "$LOCAL_ROOT" "$CACHE_ROOT"
}
trap cleanup EXIT

test "$(sha256sum "$RESUME_SHA256S" | awk '{print $1}')" = "$RESUME_SHA256S_SHA256"
(cd "$RESUME" && sha256sum --check --quiet "$RESUME_SHA256S")
(cd "$SOURCE" && sha256sum --check --quiet SOURCE_SHA256SUMS)
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA GeForce RTX 5090$')" -eq 4

mkdir -p "$ROOT" /ssd/scxi253/vision1b-research/logs
cp -a "$SOURCE/." "$LOCAL_ROOT/source/"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm \
  --format=csv -l 2 > "$ROOT/gpu.csv" 2>&1 &
monitor_pid=$!

export PYTHONPATH="/ssd/scxi253/vision1b-research/deps/pillow:$LOCAL_ROOT/source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export OMP_NUM_THREADS=1
unset CUDA_DEVICE_MAX_CONNECTIONS || true

"$PYTHON" - <<'PY'
import torch._inductor.config as config

assert config.autotune_in_subproc is True
PY

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$LOCAL_ROOT/source/train_dino_real.py" \
  --manifest "$DATA/manifest.jsonl" \
  --protocol "$PROTOCOL" \
  --acquisition-receipt "$DATA/acquisition-receipt.json" \
  --continuation-protocol "$CONTINUATION" \
  --resume "$RESUME" \
  --output-dir "$ROOT" \
  --steps 1000 \
  --batch-per-rank 112 \
  --workers-per-rank 2 \
  --save-every 250 \
  --warmup-steps 10 \
  --compile-mode max-autotune-no-cudagraphs \
  --precision fp16 \
  --loss-scale 1024 \
  --peak-dense-tflops-per-gpu 209.5

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=
find "$ROOT" -path "$ROOT/checkpoints" -prune -o -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > "$ROOT/SHA256SUMS"
