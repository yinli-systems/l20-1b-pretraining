#!/bin/bash
#SBATCH --job-name=top2-fast-fullpass
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/top2-fast-fullpass-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/top2-fast-fullpass-%j.err

set -euo pipefail
SOURCE="${SOURCE:?set SOURCE to an immutable source directory}"
SCATTERMOE_SOURCE="${SCATTERMOE_SOURCE:?set SCATTERMOE_SOURCE to the pinned checkout}"
PYTHON="${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}"
LIGER_WHEEL="${LIGER_WHEEL:-/ssd/scxi253/l20-1b-posttrain-20260913-v1/wheels/liger-kernel-0.8.2/liger_kernel-0.8.2-py3-none-any.whl}"
LIGER_SHA256=84c0a7bc9bf4d4cf8ea5ba89ff84d28686afc94215b220851d9f57dc87852741
RUN_NAME="${RUN_NAME:-proxy-top2-scattermoe-liger-mb32-fullpass-s20260917-v1}"
TARGET_TOKENS="${TARGET_TOKENS:-8376643584}"
RUN_ROOT="${RUN_ROOT:-/ssd/scxi253/cvcr-research/runs}"
RUN="$RUN_ROOT/$RUN_NAME"
DATA_ROOT=/ssd/scxi253/pretrain500m-20260912-v1/data/fineweb-edu-formal-v1
LOCAL_ROOT="$(mktemp -d "/tmp/top2-fast-fullpass-${SLURM_JOB_ID}-XXXXXX")"
cleanup() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup EXIT

mkdir -p "$RUN" /ssd/scxi253/cvcr-research/logs
test "$(sha256sum "$LIGER_WHEEL" | awk '{print $1}')" = "$LIGER_SHA256"
test "$(git -C "$SCATTERMOE_SOURCE" rev-parse HEAD)" = \
  47b5e1502e5a10e82c8e5945d761b877849871e7
"$PYTHON" - "$DATA_ROOT/manifest.json" "$TARGET_TOKENS" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
target = int(sys.argv[2])
assert manifest["status"] == "FROZEN_VERIFIED_PACK"
assert manifest["block_size"] == 2049
assert 0 < target <= int(manifest["unique_prediction_tokens"])
PY

resume_args=()
if [[ -f "$RUN/resume.pt" ]]; then
  test -f "$RUN/resume.sha256"
  (cd "$RUN" && sha256sum -c resume.sha256)
  resume_args+=(--resume)
elif [[ -e "$RUN/run-manifest.json" || -e "$RUN/metrics.jsonl" ]]; then
  echo "refusing to restart a non-empty run without a verified checkpoint" >&2
  exit 73
fi

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

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m cvcr_moe.train \
  --config "$LOCAL_ROOT/source/configs/model/oracle_top2_scattermoe_liger.json" \
  --data-dir "$DATA_ROOT/train-npy" \
  --val-dir "$DATA_ROOT/val-npy" \
  --output-dir "$RUN" \
  --target-tokens "$TARGET_TOKENS" \
  --microbatch 32 --accumulation 2 \
  --peak-lr 0.0006 --warmup-tokens 20000000 \
  --validate-every 1000 --validation-batches 8 --validation-microbatch 8 \
  --save-every 1000 --seed 20260917 \
  --compile --compile-mode default \
  --dense-bf16-tflops-per-gpu 209.5 \
  --rated-dense-bf16-tflops-per-gpu 165.2 \
  --minimum-active-mfu 0.0 --mfu-grace-steps 20 \
  "${resume_args[@]}"

status="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN/training-status.json")"
if [[ "$status" == "COMPLETED" ]]; then
  find "$RUN" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    > "$RUN/SHA256SUMS"
elif [[ "$status" != "CHECKPOINTED_STOP" ]]; then
  echo "unexpected terminal training status: $status" >&2
  exit 74
fi
