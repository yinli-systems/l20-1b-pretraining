#!/bin/bash
#SBATCH --job-name=cvcr-liger-frontier
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:35:00
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/liger-frontier-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/liger-frontier-%j.err

set -euo pipefail
SOURCE="${SOURCE:?set SOURCE to the immutable candidate source directory}"
SCATTERMOE_SOURCE="${SCATTERMOE_SOURCE:?set SCATTERMOE_SOURCE to the pinned checkout}"
PYTHON="${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}"
LIGER_WHEEL="${LIGER_WHEEL:-/ssd/scxi253/l20-1b-posttrain-20260913-v1/wheels/liger-kernel-0.8.2/liger_kernel-0.8.2-py3-none-any.whl}"
LIGER_SHA256=84c0a7bc9bf4d4cf8ea5ba89ff84d28686afc94215b220851d9f57dc87852741
RUN_ROOT="${RUN_ROOT:-/ssd/scxi253/cvcr-research/liger-microbatch-frontier}"
MODEL_CONFIG="${MODEL_CONFIG:-oracle_top2_scattermoe_liger.json}"
ROOT="$RUN_ROOT/$SLURM_JOB_ID"
LOCAL_ROOT="$(mktemp -d "/tmp/cvcr-liger-frontier-${SLURM_JOB_ID}-XXXXXX")"
cleanup() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup EXIT
mkdir -p "$ROOT" /ssd/scxi253/cvcr-research/logs
test "$(sha256sum "$LIGER_WHEEL" | awk '{print $1}')" = "$LIGER_SHA256"
cp -a "$SOURCE/." "$LOCAL_ROOT/source/"
cp -a "$SCATTERMOE_SOURCE/." "$LOCAL_ROOT/scattermoe/"
"$PYTHON" -m pip install --quiet --no-deps --target "$LOCAL_ROOT/liger" "$LIGER_WHEEL"
export PYTHONPATH="$LOCAL_ROOT/source:$LOCAL_ROOT/scattermoe:$LOCAL_ROOT/liger"
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_ROOT/inductor"
export TRITON_CACHE_DIR="$LOCAL_ROOT/triton"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4
printf 'label\tmicrobatch\taccumulation\texit_code\n' > "$ROOT/status.tsv"

run_cell() {
  local label="$1"
  local microbatch="$2"
  local accumulation="$3"
  local run="$ROOT/$label"
  mkdir -p "$run"
  set +e
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m cvcr_moe.train \
    --config "$LOCAL_ROOT/source/configs/model/$MODEL_CONFIG" \
    --data-dir /ssd/scxi253/pretrain500m-20260912-v1/data/fineweb-edu-formal-v1/train-npy \
    --val-dir /ssd/scxi253/pretrain500m-20260912-v1/data/fineweb-edu-formal-v1/val-npy \
    --output-dir "$run" --target-tokens 2000000000 --max-steps 60 \
    --microbatch "$microbatch" --accumulation "$accumulation" \
    --peak-lr 0.0006 --validate-every 60 --validation-batches 1 \
    --validation-microbatch 8 --save-every 0 --no-save-final \
    --compile --compile-mode default \
    --dense-bf16-tflops-per-gpu 209.5 \
    --rated-dense-bf16-tflops-per-gpu 165.2 \
    --minimum-active-mfu 0.0 --mfu-grace-steps 20
  local code=$?
  set -e
  printf '%s\t%s\t%s\t%s\n' "$label" "$microbatch" "$accumulation" "$code" \
    >> "$ROOT/status.tsv"
  if test "$code" -eq 0; then
    "$PYTHON" "$LOCAL_ROOT/source/scripts/summarize_metrics.py" "$run" \
      --warmup 20 --output "$ROOT/$label-summary.json"
    return 0
  fi
  return "$code"
}

run_cell A-mb16 16 4
if run_cell B-mb32 32 2; then
  run_cell C-mb64 64 1 || true
fi
find "$ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
  > "$ROOT/SHA256SUMS"
