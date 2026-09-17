#!/bin/bash
#SBATCH --job-name=moe7b-fsdp2-4x5090
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --open-mode=append
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/moe7b-fsdp2-4x5090-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/moe7b-fsdp2-4x5090-%j.err

set -euo pipefail
SOURCE="${SOURCE:?set SOURCE to an immutable source directory}"
SCATTERMOE_SOURCE="${SCATTERMOE_SOURCE:?set SCATTERMOE_SOURCE to the pinned checkout}"
PYTHON="${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}"
LIGER_WHEEL="${LIGER_WHEEL:-/ssd/scxi253/l20-1b-posttrain-20260913-v1/wheels/liger-kernel-0.8.2/liger_kernel-0.8.2-py3-none-any.whl}"
LIGER_SHA256=84c0a7bc9bf4d4cf8ea5ba89ff84d28686afc94215b220851d9f57dc87852741
SCATTERMOE_COMMIT=47b5e1502e5a10e82c8e5945d761b877849871e7
RUN_NAME="${RUN_NAME:-moe7b-top2-scattermoe-fsdp2-4x5090-s20260917-v1}"
TARGET_TOKENS="${TARGET_TOKENS:-8376643584}"
ACCUMULATION="${ACCUMULATION:-16}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
REQUIRED_DEVICE="${REQUIRED_DEVICE:-NVIDIA GeForce RTX 5090}"
RUN_ROOT="${RUN_ROOT:-/ssd/scxi253/cvcr-research/runs}"
RUN="$RUN_ROOT/$RUN_NAME"
DATA_ROOT=/ssd/scxi253/pretrain500m-20260912-v1/data/fineweb-edu-formal-v1
LOCAL_ROOT="$(mktemp -d "/tmp/moe7b-fsdp2-${SLURM_JOB_ID}-XXXXXX")"
child_pid=""
signal_seen=0

cleanup() {
  chmod -R u+w "$LOCAL_ROOT" 2>/dev/null || true
  rm -rf -- "$LOCAL_ROOT"
}
forward_signal() {
  signal_seen=1
  if [[ -n "$child_pid" ]]; then
    pkill -USR1 -P "$child_pid" 2>/dev/null || true
    kill -USR1 "$child_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap forward_signal USR1 TERM INT

mkdir -p "$RUN_ROOT" /ssd/scxi253/cvcr-research/logs
test "$(sha256sum "$LIGER_WHEEL" | awk '{print $1}')" = "$LIGER_SHA256"
test "$(git -C "$SCATTERMOE_SOURCE" rev-parse HEAD)" = "$SCATTERMOE_COMMIT"
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
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export SCATTERMOE_COMMIT LIGER_WHEEL_SHA256="$LIGER_SHA256"

common_args=(
  -m cvcr_moe.train_fsdp
  --config "$LOCAL_ROOT/source/configs/model/moe_7b_top2_scattermoe_liger.json"
  --data-dir "$DATA_ROOT/train-npy"
  --val-dir "$DATA_ROOT/val-npy"
  --output-dir "$RUN"
  --target-tokens "$TARGET_TOKENS"
  --microbatch 1 --accumulation "$ACCUMULATION"
  --peak-lr 0.0004 --warmup-tokens 200000000
  --bf16-optimizer-states
  --activation-checkpointing
  --log-every 10
  --checkpoint-every 5000 --keep-checkpoints 2
  --validate-every 10000 --validation-batches 4
  --seed 20260917
  --rated-dense-bf16-tflops-per-gpu 209.5
  --minimum-causal-mfu 0.0 --mfu-grace-steps 20
  --required-device "$REQUIRED_DEVICE"
)

run_training() {
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
    "${common_args[@]}" "$@" &
  child_pid=$!
  local rc=0
  wait "$child_pid" || rc=$?
  child_pid=""
  if [[ "$signal_seen" == 1 ]]; then
    return 0
  fi
  return "$rc"
}

read_status() {
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$RUN/training-status.json"
}

if [[ -f "$RUN/latest.json" ]]; then
  run_training --resume
else
  if [[ -d "$RUN" && -n "$(find "$RUN" -mindepth 1 -print -quit)" ]]; then
    echo "refusing to restart a non-empty 7B run without a verified latest checkpoint" >&2
    exit 73
  fi
  # The first optimizer step is part of the real run. Commit it, reconstruct
  # the model, and verify a full model/Adam/RNG resume before training onward.
  run_training --steps-this-launch 1 --log-every 1 --checkpoint-every 0
  if [[ "$signal_seen" == 1 ]]; then
    /usr/bin/scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  test "$(read_status)" = CHECKPOINTED_SEGMENT
  run_training --resume
fi

status="$(read_status)"
if [[ "$status" == COMPLETED ]]; then
  find "$RUN" -maxdepth 2 -type f ! -name SHA256SUMS -print0 | sort -z | \
    xargs -0 sha256sum > "$RUN/SHA256SUMS"
elif [[ "$status" == CHECKPOINTED_STOP || "$signal_seen" == 1 ]]; then
  /usr/bin/scontrol requeue "$SLURM_JOB_ID"
else
  echo "unexpected terminal training status: $status" >&2
  exit 74
fi
