#!/bin/bash
#SBATCH --job-name=cvcr-proxy-screen
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/proxy-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/proxy-%j.err

set -euo pipefail
SOURCE="${SOURCE:-/ssd/scxi253/cvcr-research/source}"
PYTHON="${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}"
SCATTERMOE_SOURCE="${SCATTERMOE_SOURCE:-}"
MODEL_CONFIG="${MODEL_CONFIG:-proxy_top2.json}"
RUN_NAME="${RUN_NAME:-screen-${MODEL_CONFIG%.json}-${SLURM_JOB_ID}}"
COMPILE_MODE="${COMPILE_MODE:-default}"
COMPILE="${COMPILE:-1}"
MAX_STEPS="${MAX_STEPS:-20}"
MICROBATCH="${MICROBATCH:-1}"
ACCUMULATION="${ACCUMULATION:-1}"
RUN=/ssd/scxi253/cvcr-research/runs/$RUN_NAME
LOCAL_ROOT="$(mktemp -d "/tmp/cvcr-proxy-${SLURM_JOB_ID}-XXXXXX")"
cleanup() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup EXIT
mkdir -p "$RUN"
cp -a "$SOURCE/." "$LOCAL_ROOT/source/"
if [[ -n "$SCATTERMOE_SOURCE" ]]; then
  cp -a "$SCATTERMOE_SOURCE/." "$LOCAL_ROOT/scattermoe/"
  export PYTHONPATH="$LOCAL_ROOT/source:$LOCAL_ROOT/scattermoe"
else
  export PYTHONPATH="$LOCAL_ROOT/source"
fi
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_ROOT/inductor"
export TRITON_CACHE_DIR="$LOCAL_ROOT/triton"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4

compile_args=()
if [[ "$COMPILE" == "1" ]]; then
  compile_args+=(--compile --compile-mode "$COMPILE_MODE")
fi

"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  -m cvcr_moe.train \
  --config "$LOCAL_ROOT/source/configs/model/$MODEL_CONFIG" \
  --data-dir /ssd/scxi253/pretrain500m-20260912-v1/data/fineweb-edu-formal-v1/train-npy \
  --val-dir /ssd/scxi253/pretrain500m-20260912-v1/data/fineweb-edu-formal-v1/val-npy \
  --output-dir "$RUN" \
  --target-tokens 2000000000 \
  --max-steps "$MAX_STEPS" \
  --microbatch "$MICROBATCH" \
  --accumulation "$ACCUMULATION" \
  --peak-lr 0.0006 \
  --validate-every "$MAX_STEPS" \
  --validation-batches 1 \
  --save-every 0 \
  --no-save-final \
  "${compile_args[@]}" \
  --dense-bf16-tflops-per-gpu 209.5 \
  --rated-dense-bf16-tflops-per-gpu 165.2 \
  --minimum-active-mfu 0.0 \
  --mfu-grace-steps 5
