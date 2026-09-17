#!/bin/bash
#SBATCH --job-name=cvcr-ddp-bucket
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:35:00
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/ddp-bucket-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/ddp-bucket-%j.err

set -euo pipefail
SOURCE="${SOURCE:?set SOURCE to the immutable candidate source directory}"
SCATTERMOE_SOURCE="${SCATTERMOE_SOURCE:?set SCATTERMOE_SOURCE to the pinned checkout}"
PYTHON="${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-/ssd/scxi253/cvcr-research/ddp-bucket}"
ROOT="$RUN_ROOT/$SLURM_JOB_ID"
LOCAL_ROOT="$(mktemp -d "/tmp/cvcr-ddp-bucket-${SLURM_JOB_ID}-XXXXXX")"
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
  local bucket="$2"
  local run="$ROOT/$label"
  mkdir -p "$run"
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m cvcr_moe.train \
    --config "$LOCAL_ROOT/source/configs/model/oracle_top2_scattermoe.json" \
    --data-dir /ssd/scxi253/pretrain500m-20260912-v1/data/fineweb-edu-formal-v1/train-npy \
    --val-dir /ssd/scxi253/pretrain500m-20260912-v1/data/fineweb-edu-formal-v1/val-npy \
    --output-dir "$run" \
    --target-tokens 2000000000 --max-steps 60 \
    --microbatch 8 --accumulation 8 --peak-lr 0.0006 \
    --validate-every 60 --validation-batches 1 --save-every 0 --no-save-final \
    --compile --compile-mode default --ddp-bucket-cap-mb "$bucket" \
    --dense-bf16-tflops-per-gpu 209.5 \
    --rated-dense-bf16-tflops-per-gpu 165.2 \
    --minimum-active-mfu 0.0 --mfu-grace-steps 20
}

run_cell A-bucket25 25
run_cell B-bucket8 8
run_cell C-bucket100 100

for label in A-bucket25 B-bucket8 C-bucket100; do
  "$PYTHON" "$LOCAL_ROOT/source/scripts/summarize_metrics.py" "$ROOT/$label" \
    --warmup 20 --output "$ROOT/$label-summary.json"
done
find "$ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
  > "$ROOT/SHA256SUMS"
