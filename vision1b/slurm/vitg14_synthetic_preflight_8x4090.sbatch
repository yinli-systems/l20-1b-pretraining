#!/bin/bash
#SBATCH --job-name=vitg14-1b-frontier
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=00:45:00
#SBATCH --output=/ssd/scxi253/vision1b-research/logs/vitg14-1b-preflight-%j.out
#SBATCH --error=/ssd/scxi253/vision1b-research/logs/vitg14-1b-preflight-%j.err

set -euo pipefail

PYTHON=/ssd/scxi253/pretraining2/runtime/venv/bin/python
SOURCE=/ssd/scxi253/vision1b-research/source
BATCH_PER_RANK=${BATCH_PER_RANK:-1}
COMPILE_MODE=${COMPILE_MODE:-max-autotune-no-cudagraphs}
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-true}
FUSED_MULTICROP=${FUSED_MULTICROP:-true}
TEACHER_RESHARD_AFTER_FORWARD=${TEACHER_RESHARD_AFTER_FORWARD:-false}
STUDENT_RESHARD_AFTER_FORWARD=${STUDENT_RESHARD_AFTER_FORWARD:-true}
STEPS=${STEPS:-12}
WARMUP_STEPS=${WARMUP_STEPS:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
SHARD_GROUP_SIZE=${SHARD_GROUP_SIZE:-$NPROC_PER_NODE}
ROOT=/ssd/scxi253/vision1b-research/runs/vitg14-1b-frontier-g${NPROC_PER_NODE}-b${BATCH_PER_RANK}-ckpt${ACTIVATION_CHECKPOINTING}-$SLURM_JOB_ID
LOCAL_ROOT=$(mktemp -d "/tmp/vitg14-1b-$SLURM_JOB_ID-XXXXXX")
monitor_pid=

cleanup() {
  if test -n "$monitor_pid"; then kill "$monitor_pid" 2>/dev/null || true; fi
  rm -rf -- "$LOCAL_ROOT"
}
trap cleanup EXIT

mkdir -p "$ROOT" /ssd/scxi253/vision1b-research/logs
cp "$SOURCE/vitg14_preflight.py" "$LOCAL_ROOT/vitg14_preflight.py"
sha256sum "$LOCAL_ROOT/vitg14_preflight.py" > "$ROOT/source.sha256"
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,clocks.sm --format=csv -l 2 > "$ROOT/gpu.csv" 2>&1 &
monitor_pid=$!

export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_ROOT/inductor"
export TRITON_CACHE_DIR="$LOCAL_ROOT/triton"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
unset CUDA_DEVICE_MAX_CONNECTIONS || true
export OMP_NUM_THREADS=4

checkpoint_flag=--activation-checkpointing
if test "$ACTIVATION_CHECKPOINTING" = false; then
  checkpoint_flag=--no-activation-checkpointing
fi
multicrop_flag=--fused-multicrop
if test "$FUSED_MULTICROP" = false; then
  multicrop_flag=--no-fused-multicrop
fi
teacher_reshard_flag=--teacher-reshard-after-forward
if test "$TEACHER_RESHARD_AFTER_FORWARD" = false; then
  teacher_reshard_flag=--no-teacher-reshard-after-forward
fi
student_reshard_flag=--student-reshard-after-forward
if test "$STUDENT_RESHARD_AFTER_FORWARD" = false; then
  student_reshard_flag=--no-student-reshard-after-forward
fi

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
  "$LOCAL_ROOT/vitg14_preflight.py" \
  --output-dir "$ROOT" \
  --steps "$STEPS" \
  --warmup-steps "$WARMUP_STEPS" \
  --batch-per-rank "$BATCH_PER_RANK" \
  --shard-group-size "$SHARD_GROUP_SIZE" \
  --global-views 2 \
  --local-views 8 \
  "$checkpoint_flag" \
  "$multicrop_flag" \
  "$teacher_reshard_flag" \
  "$student_reshard_flag" \
  --compile-mode "$COMPILE_MODE"

find "$ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$ROOT/SHA256SUMS"
