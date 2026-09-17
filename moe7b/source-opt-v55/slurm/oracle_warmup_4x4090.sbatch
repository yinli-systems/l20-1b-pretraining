#!/bin/bash
#SBATCH --job-name=cvcr-oracle-warmup
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/warmup-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/warmup-%j.err

set -euo pipefail
: "${RUN_NAME:?set a unique RUN_NAME}"
MODEL_CONFIG="${MODEL_CONFIG:-oracle_top2.json}"
TARGET_TOKENS="${TARGET_TOKENS:-200000000}"
MICROBATCH="${MICROBATCH:-8}"
ACCUMULATION="${ACCUMULATION:-8}"
COMPILE="${COMPILE:-1}"
COMPILE_MODE="${COMPILE_MODE:-default}"
SEED="${SEED:-20260916}"
SOURCE="${SOURCE:-/ssd/scxi253/cvcr-research/source}"
PYTHON="${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}"
SCATTERMOE_SOURCE="${SCATTERMOE_SOURCE:-}"
RUN=/ssd/scxi253/cvcr-research/runs/$RUN_NAME
LOCAL_ROOT="$(mktemp -d "/tmp/cvcr-warmup-${SLURM_JOB_ID}-XXXXXX")"
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
  --target-tokens "$TARGET_TOKENS" \
  --microbatch "$MICROBATCH" \
  --accumulation "$ACCUMULATION" \
  --peak-lr 0.0006 \
  --warmup-tokens 20000000 \
  --validate-every 100 \
  --validation-batches 8 \
  --save-every 250 \
  --seed "$SEED" \
  "${compile_args[@]}" \
  --dense-bf16-tflops-per-gpu 209.5 \
  --rated-dense-bf16-tflops-per-gpu 165.2 \
  --minimum-active-mfu 0.0 \
  --mfu-grace-steps 10
