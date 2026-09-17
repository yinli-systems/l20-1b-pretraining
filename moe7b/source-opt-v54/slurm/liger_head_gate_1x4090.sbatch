#!/bin/bash
#SBATCH --job-name=cvcr-liger-head
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/liger-head-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/liger-head-%j.err

set -euo pipefail
SOURCE="${SOURCE:?set SOURCE to the immutable candidate source directory}"
PYTHON="${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}"
LIGER_WHEEL="${LIGER_WHEEL:-/ssd/scxi253/l20-1b-posttrain-20260913-v1/wheels/liger-kernel-0.8.2/liger_kernel-0.8.2-py3-none-any.whl}"
LIGER_SHA256=84c0a7bc9bf4d4cf8ea5ba89ff84d28686afc94215b220851d9f57dc87852741
RUN_ROOT="${RUN_ROOT:-/ssd/scxi253/cvcr-research/liger-head-gates}"
RUN="$RUN_ROOT/$SLURM_JOB_ID"
LOCAL_ROOT="$(mktemp -d "/tmp/cvcr-liger-${SLURM_JOB_ID}-XXXXXX")"
cleanup() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup EXIT
mkdir -p "$RUN" /ssd/scxi253/cvcr-research/logs
test "$(sha256sum "$LIGER_WHEEL" | awk '{print $1}')" = "$LIGER_SHA256"
cp -a "$SOURCE/." "$LOCAL_ROOT/source/"
"$PYTHON" -m pip install --quiet --no-deps --target "$LOCAL_ROOT/liger" "$LIGER_WHEEL"
export PYTHONPATH="$LOCAL_ROOT/source:$LOCAL_ROOT/liger"
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export TRITON_CACHE_DIR="$LOCAL_ROOT/triton"
export OMP_NUM_THREADS=6

"$PYTHON" "$LOCAL_ROOT/source/scripts/validate_liger_head.py" \
  --tokens 8192 --hidden 768 --vocab 50280 --warmup 3 --steps 10 \
  | tee "$RUN/result.json"
sha256sum "$RUN/result.json" > "$RUN/SHA256SUMS"
