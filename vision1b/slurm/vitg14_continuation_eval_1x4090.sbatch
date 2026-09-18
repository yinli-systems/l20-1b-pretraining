#!/bin/bash
#SBATCH --job-name=vitg14-cont-eval
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/vision1b-research/logs/vitg14-cont-eval-%j.out
#SBATCH --error=/ssd/scxi253/vision1b-research/logs/vitg14-cont-eval-%j.err

set -euo pipefail

PYTHON=/ssd/scxi253/pretraining2/runtime/venv/bin/python
BASE=/ssd/scxi253/vision1b-research
SOURCE=$BASE/source-eval-20260918-v2
RUN=$BASE/runs/vision1b-25k-cont1000-r1-1598870
DATA_ROOT=$BASE/data/downstream

EVAL_STEP=${EVAL_STEP:?EVAL_STEP must be 500 or 1000}
EVAL_MODE=${EVAL_MODE:?EVAL_MODE must be smoke or full}
case "$EVAL_STEP" in
  500) STEP_DIR=step-000500 ;;
  1000) STEP_DIR=step-001000 ;;
  *) echo "unsupported EVAL_STEP=$EVAL_STEP" >&2; exit 2 ;;
esac
case "$EVAL_MODE" in
  smoke)
    COMPILE_MODE=none
    MAX_TRAIN=256
    MAX_TEST=128
    ;;
  full)
    COMPILE_MODE=max-autotune-no-cudagraphs
    MAX_TRAIN=0
    MAX_TEST=0
    ;;
  *) echo "unsupported EVAL_MODE=$EVAL_MODE" >&2; exit 2 ;;
esac

CHECKPOINT=$RUN/checkpoints/$STEP_DIR
CHECKPOINT_SHA256_RECEIPT=$RUN/checkpoints/${STEP_DIR}.SHA256SUMS
PROTOCOL=$SOURCE/frozen_eval_step${EVAL_STEP}_protocol.json
OUTPUT_ROOT=$BASE/evaluations/vitg14-cont-step${EVAL_STEP}-${EVAL_MODE}-$SLURM_JOB_ID
LOCAL_ROOT=$(mktemp -d "/tmp/vitg14-cont-eval-$SLURM_JOB_ID-XXXXXX")
CACHE_ROOT="/tmp/vision1b-cont-eval-cache-$USER-$SLURM_JOB_ID"
monitor_pid=

stop_monitor() {
  if test -n "$monitor_pid"; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=
  fi
}

cleanup() {
  stop_monitor
  rm -rf -- "$LOCAL_ROOT" "$CACHE_ROOT"
}
trap cleanup EXIT

mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT" "$BASE/logs"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256SUMS)

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if test "${#gpu_names[@]}" -ne 1 || test "${gpu_names[0]}" != "NVIDIA GeForce RTX 4090"; then
  printf 'expected one NVIDIA GeForce RTX 4090, got %s\n' "${gpu_names[*]:-none}" >&2
  exit 3
fi
nvidia-smi -q >/dev/null

cp "$SOURCE/vitg14_preflight.py" "$SOURCE/train_dino_real.py" \
  "$SOURCE/evaluate_frozen_features.py" "$LOCAL_ROOT/"
sha256sum "$LOCAL_ROOT"/*.py "$PROTOCOL" > "$OUTPUT_ROOT/source.sha256"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,clocks.sm \
  --format=csv -l 2 > "$OUTPUT_ROOT/gpu.csv" 2>&1 &
monitor_pid=$!

export PYTHONPATH=$BASE/deps/pillow${PYTHONPATH:+:$PYTHONPATH}
export PYTHONPYCACHEPREFIX=$LOCAL_ROOT/pycache
export TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT/inductor
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export OMP_NUM_THREADS=1

extra_args=()
if test "$MAX_TRAIN" -gt 0; then extra_args+=(--max-train "$MAX_TRAIN"); fi
if test "$MAX_TEST" -gt 0; then extra_args+=(--max-test "$MAX_TEST"); fi

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
  "$LOCAL_ROOT/evaluate_frozen_features.py" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256-receipt "$CHECKPOINT_SHA256_RECEIPT" \
  --protocol "$PROTOCOL" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_ROOT" \
  --batch-per-rank 128 \
  --compile-mode "$COMPILE_MODE" \
  --variants student \
  "${extra_args[@]}"

"$PYTHON" - "$OUTPUT_ROOT/results.json" "$EVAL_MODE" <<'PY'
import json
import math
import sys

path, mode = sys.argv[1:]
result = json.load(open(path))
if result.get("status") != "COMPLETE" or result.get("world_size") != 1:
    raise SystemExit("evaluation did not produce a complete one-rank result")
if set(result.get("results", {})) != {"student"}:
    raise SystemExit("evaluation result must contain only the student variant")
expected = (256, 128) if mode == "smoke" else (60000, 10000)
for dataset in ("fashion_mnist", "mnist"):
    row = result["results"]["student"].get(dataset, {})
    if (row.get("train_examples"), row.get("test_examples")) != expected:
        raise SystemExit(f"unexpected sample counts for {dataset}")
    for metric in (row.get("knn", {}).get("top1"), row.get("ridge_probe", {}).get("top1")):
        if not isinstance(metric, (int, float)) or not math.isfinite(metric):
            raise SystemExit(f"nonfinite primary score for {dataset}")
PY

stop_monitor
find "$OUTPUT_ROOT" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/SHA256SUMS"
(cd "$OUTPUT_ROOT" && sha256sum -c SHA256SUMS)
