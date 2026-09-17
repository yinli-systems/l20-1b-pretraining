#!/bin/bash
#SBATCH --job-name=moe7b-ep4-screen
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/moe7b-ep4-screen-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/moe7b-ep4-screen-%j.err

set -euo pipefail
SOURCE="${SOURCE:?set SOURCE to an immutable source directory}"
SCATTERMOE_SOURCE="${SCATTERMOE_SOURCE:?set SCATTERMOE_SOURCE to the pinned checkout}"
PYTHON="${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}"
LIGER_WHEEL="${LIGER_WHEEL:-/ssd/scxi253/l20-1b-posttrain-20260913-v1/wheels/liger-kernel-0.8.2/liger_kernel-0.8.2-py3-none-any.whl}"
LIGER_SHA256=84c0a7bc9bf4d4cf8ea5ba89ff84d28686afc94215b220851d9f57dc87852741
SCATTERMOE_COMMIT=47b5e1502e5a10e82c8e5945d761b877849871e7
MICROBATCH="${MICROBATCH:-4}"
ACCUMULATION="${ACCUMULATION:-4}"
MAX_STEPS="${MAX_STEPS:-12}"
RUN_ROOT="${RUN_ROOT:-/ssd/scxi253/cvcr-research/moe7b-expert-parallel-frontier}"
ROOT="$RUN_ROOT/job-${SLURM_JOB_ID}-ep4-mb${MICROBATCH}-acc${ACCUMULATION}"
RUN="$ROOT/train"
DATA_ROOT=/ssd/scxi253/pretrain500m-20260912-v1/data/fineweb-edu-formal-v1
LOCAL_ROOT="$(mktemp -d "/tmp/moe7b-ep4-${SLURM_JOB_ID}-XXXXXX")"
monitor_pid=""

cleanup() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  chmod -R u+w "$LOCAL_ROOT" 2>/dev/null || true
  rm -rf -- "$LOCAL_ROOT"
}
trap cleanup EXIT

case "$MICROBATCH:$ACCUMULATION" in
  1:16|2:8|4:4|8:2|16:1) ;;
  *) echo "layout must preserve sixteen sequences per rank per optimizer step" >&2; exit 64 ;;
esac
test "$((MICROBATCH * ACCUMULATION))" -eq 16
test "$MAX_STEPS" -ge 8
test "$(sha256sum "$LIGER_WHEEL" | awk '{print $1}')" = "$LIGER_SHA256"
test "$(git -C "$SCATTERMOE_SOURCE" rev-parse HEAD)" = "$SCATTERMOE_COMMIT"
test -f "$SOURCE/SOURCE_SHA256SUMS"
(cd "$SOURCE" && sha256sum --check --quiet SOURCE_SHA256SUMS)

mkdir -p "$ROOT" /ssd/scxi253/cvcr-research/logs
cp -a "$SOURCE/." "$LOCAL_ROOT/source/"
cp -a "$SCATTERMOE_SOURCE/." "$LOCAL_ROOT/scattermoe/"
"$PYTHON" -m pip install --quiet --no-deps --target "$LOCAL_ROOT/liger" "$LIGER_WHEEL"
export PYTHONPATH="$LOCAL_ROOT/source:$LOCAL_ROOT/scattermoe:$LOCAL_ROOT/liger"
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_ROOT/inductor"
export TRITON_CACHE_DIR="$LOCAL_ROOT/triton"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export SCATTERMOE_COMMIT LIGER_WHEEL_SHA256="$LIGER_SHA256"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -l 2 > "$ROOT/gpu-telemetry.csv" &
monitor_pid=$!

set +e
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m cvcr_moe.train_ep \
  --config "$LOCAL_ROOT/source/configs/model/moe_7b_top2_scattermoe_liger.json" \
  --data-dir "$DATA_ROOT/train-npy" \
  --val-dir "$DATA_ROOT/val-npy" \
  --output-dir "$RUN" \
  --target-tokens 8376643584 \
  --microbatch "$MICROBATCH" --accumulation "$ACCUMULATION" \
  --max-steps "$MAX_STEPS" \
  --peak-lr 0.0004 --warmup-tokens 200000000 \
  --log-every 1 --seed 20260917 \
  --ddp-bucket-cap-mb 100 \
  --rated-dense-bf16-tflops-per-gpu 209.5 \
  --required-device "NVIDIA GeForce RTX 5090"
training_code=$?
set -e

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=""
printf '%s\n' "$training_code" > "$ROOT/training-exit-code.txt"
if [[ "$training_code" -eq 0 ]]; then
  "$PYTHON" "$LOCAL_ROOT/source/scripts/summarize_expert_parallel.py" "$RUN" \
    --warmup-records 2 --minimum-mfu 0.50 --minimum-headroom-gib 2 \
    --output "$ROOT/summary.json"
fi
find "$ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > "$ROOT/SHA256SUMS"
exit "$training_code"
