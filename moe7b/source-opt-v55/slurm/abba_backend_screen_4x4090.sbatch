#!/bin/bash
#SBATCH --job-name=cvcr-backend-abba
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/backend-abba-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/backend-abba-%j.err

set -euo pipefail
SOURCE="${SOURCE:?set SOURCE to the immutable candidate source directory}"
SCATTERMOE_SOURCE="${SCATTERMOE_SOURCE:?set SCATTERMOE_SOURCE to the pinned checkout}"
PYTHON="${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-/ssd/scxi253/cvcr-research/backend-abba}"
MAX_STEPS="${MAX_STEPS:-120}"
MICROBATCH="${MICROBATCH:-8}"
ACCUMULATION="${ACCUMULATION:-8}"
ROOT="$RUN_ROOT/$SLURM_JOB_ID"
LOCAL_ROOT="$(mktemp -d "/tmp/cvcr-abba-${SLURM_JOB_ID}-XXXXXX")"
cleanup() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup EXIT
mkdir -p "$ROOT" /ssd/scxi253/cvcr-research/logs
cp -a "$SOURCE/." "$LOCAL_ROOT/source/"
cp -a "$SCATTERMOE_SOURCE/." "$LOCAL_ROOT/scattermoe/"
export PYTHONPATH="$LOCAL_ROOT/source:$LOCAL_ROOT/scattermoe"
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_ROOT/inductor"
export TRITON_CACHE_DIR="$LOCAL_ROOT/triton"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4

run_cell() {
  local label="$1"
  local model_config="$2"
  local run="$ROOT/$label"
  mkdir -p "$run"
  "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    -m cvcr_moe.train \
    --config "$LOCAL_ROOT/source/configs/model/$model_config" \
    --data-dir /ssd/scxi253/pretrain500m-20260912-v1/data/fineweb-edu-formal-v1/train-npy \
    --val-dir /ssd/scxi253/pretrain500m-20260912-v1/data/fineweb-edu-formal-v1/val-npy \
    --output-dir "$run" \
    --target-tokens 2000000000 \
    --max-steps "$MAX_STEPS" \
    --microbatch "$MICROBATCH" \
    --accumulation "$ACCUMULATION" \
    --peak-lr 0.0006 \
    --validate-every "$MAX_STEPS" \
    --validation-batches 1 \
    --save-every 0 \
    --no-save-final \
    --compile --compile-mode default \
    --dense-bf16-tflops-per-gpu 209.5 \
    --rated-dense-bf16-tflops-per-gpu 165.2 \
    --minimum-active-mfu 0.0 \
    --mfu-grace-steps 20
}

run_cell A1-native oracle_top2.json
run_cell B1-scattermoe oracle_top2_scattermoe.json
run_cell B2-scattermoe oracle_top2_scattermoe.json
run_cell A2-native oracle_top2.json

"$PYTHON" "$LOCAL_ROOT/source/scripts/compare_backend_abba.py" \
  --a1 "$ROOT/A1-native" \
  --b1 "$ROOT/B1-scattermoe" \
  --b2 "$ROOT/B2-scattermoe" \
  --a2 "$ROOT/A2-native" \
  --warmup 20 \
  --output "$ROOT/summary.json"

find "$ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
  > "$ROOT/SHA256SUMS"
