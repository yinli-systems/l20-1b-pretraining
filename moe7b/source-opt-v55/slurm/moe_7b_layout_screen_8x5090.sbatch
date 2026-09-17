#!/bin/bash
#SBATCH --job-name=moe7b-layout-8x5090
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/moe7b-layout-8x5090-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/moe7b-layout-8x5090-%j.err

set -euo pipefail
SOURCE="${SOURCE:?set SOURCE to an immutable source directory}"
SCATTERMOE_SOURCE="${SCATTERMOE_SOURCE:?set SCATTERMOE_SOURCE to the pinned checkout}"
PYTHON="${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}"
LIGER_WHEEL="${LIGER_WHEEL:-/ssd/scxi253/l20-1b-posttrain-20260913-v1/wheels/liger-kernel-0.8.2/liger_kernel-0.8.2-py3-none-any.whl}"
LIGER_SHA256=84c0a7bc9bf4d4cf8ea5ba89ff84d28686afc94215b220851d9f57dc87852741
SCATTERMOE_COMMIT=47b5e1502e5a10e82c8e5945d761b877849871e7
MICROBATCH="${MICROBATCH:?set MICROBATCH}"
ACCUMULATION="${ACCUMULATION:?set ACCUMULATION}"
MAX_STEPS="${MAX_STEPS:-12}"
BASELINE_THROUGHPUT="${BASELINE_THROUGHPUT:-11260}"
BASELINE_CAUSAL_MFU="${BASELINE_CAUSAL_MFU:-0.05085}"
RUN_ROOT="${RUN_ROOT:-/ssd/scxi253/cvcr-research/moe7b-layout-frontier}"
ROOT="$RUN_ROOT/job-${SLURM_JOB_ID}-mb${MICROBATCH}-acc${ACCUMULATION}"
RUN="$ROOT/train"
DATA_ROOT=/ssd/scxi253/pretrain500m-20260912-v1/data/fineweb-edu-formal-v1
LOCAL_ROOT="$(mktemp -d "/tmp/moe7b-layout-${SLURM_JOB_ID}-XXXXXX")"
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
  1:8|2:4|4:2|8:1) ;;
  *) echo "layout must preserve eight sequences per rank per optimizer step" >&2; exit 64 ;;
esac
test "$((MICROBATCH * ACCUMULATION))" -eq 8
test "$MAX_STEPS" -ge 8
test "$(sha256sum "$LIGER_WHEEL" | awk '{print $1}')" = "$LIGER_SHA256"
test "$(git -C "$SCATTERMOE_SOURCE" rev-parse HEAD)" = "$SCATTERMOE_COMMIT"
test -f "$SOURCE/SOURCE_SHA256SUMS"
(cd "$SOURCE" && sha256sum --check --quiet SOURCE_SHA256SUMS)
"$PYTHON" - "$DATA_ROOT/manifest.json" <<'PY'
import json
import pathlib
import sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert manifest["status"] == "FROZEN_VERIFIED_PACK"
assert manifest["block_size"] == 2049
assert int(manifest["unique_prediction_tokens"]) >= 8_376_643_584
PY

mkdir -p "$ROOT" /ssd/scxi253/cvcr-research/logs
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
export SOURCE MICROBATCH ACCUMULATION MAX_STEPS BASELINE_THROUGHPUT BASELINE_CAUSAL_MFU

"$PYTHON" - "$ROOT/screen-identity.json" <<PY
import json, os, pathlib, time
pathlib.Path(os.environ["SOURCE"]).resolve()
value = {
    "schema": "cvcr-moe-7b-layout-screen-launch-v1",
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "source": os.environ["SOURCE"],
    "microbatch": int(os.environ["MICROBATCH"]),
    "accumulation": int(os.environ["ACCUMULATION"]),
    "max_steps": int(os.environ["MAX_STEPS"]),
    "baseline_tokens_per_second": float(os.environ["BASELINE_THROUGHPUT"]),
    "baseline_causal_mfu": float(os.environ["BASELINE_CAUSAL_MFU"]),
    "unix": time.time(),
}
pathlib.Path("$ROOT/screen-identity.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -l 2 > "$ROOT/gpu-telemetry.csv" &
monitor_pid=$!

set +e
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m cvcr_moe.train_fsdp \
  --config "$LOCAL_ROOT/source/configs/model/moe_7b_top2_scattermoe_liger.json" \
  --data-dir "$DATA_ROOT/train-npy" \
  --val-dir "$DATA_ROOT/val-npy" \
  --output-dir "$RUN" \
  --target-tokens 8376643584 \
  --max-steps "$MAX_STEPS" \
  --run-role bounded_feasibility_screen \
  --microbatch "$MICROBATCH" --accumulation "$ACCUMULATION" \
  --peak-lr 0.0004 --warmup-tokens 200000000 \
  --bf16-optimizer-states \
  --activation-checkpointing \
  --sync-every-microbatch \
  --log-every 1 \
  --checkpoint-every 0 --keep-checkpoints 1 \
  --validate-every 0 --validation-batches 1 \
  --no-save-final \
  --seed 20260917 \
  --rated-dense-bf16-tflops-per-gpu 209.5 \
  --minimum-causal-mfu 0.0 --mfu-grace-steps 3 \
  --required-device "NVIDIA GeForce RTX 5090"
training_code=$?
set -e

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=""
printf '%s\n' "$training_code" > "$ROOT/training-exit-code.txt"
if [[ "$training_code" -eq 0 ]]; then
  "$PYTHON" "$LOCAL_ROOT/source/scripts/summarize_fsdp_layout.py" "$RUN" \
    --warmup-records 2 \
    --baseline-throughput "$BASELINE_THROUGHPUT" \
    --baseline-causal-mfu "$BASELINE_CAUSAL_MFU" \
    --output "$ROOT/summary.json"
fi
find "$ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > "$ROOT/SHA256SUMS"
exit "$training_code"
