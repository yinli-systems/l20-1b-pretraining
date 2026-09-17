#!/bin/bash
#SBATCH --job-name=cvcr-liger-parity
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/liger-parity-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/liger-parity-%j.err

set -euo pipefail
SOURCE="${SOURCE:?set SOURCE to the immutable candidate source directory}"
SCATTERMOE_SOURCE="${SCATTERMOE_SOURCE:?set SCATTERMOE_SOURCE to the pinned checkout}"
PYTHON="${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}"
LIGER_WHEEL="${LIGER_WHEEL:-/ssd/scxi253/l20-1b-posttrain-20260913-v1/wheels/liger-kernel-0.8.2/liger_kernel-0.8.2-py3-none-any.whl}"
LIGER_SHA256=84c0a7bc9bf4d4cf8ea5ba89ff84d28686afc94215b220851d9f57dc87852741
RUN_ROOT="${RUN_ROOT:-/ssd/scxi253/cvcr-research/liger-update-parity}"
CANDIDATE_MICROBATCH="${CANDIDATE_MICROBATCH:-32}"
CANDIDATE_ACCUMULATION="${CANDIDATE_ACCUMULATION:-2}"
REFERENCE_CONFIG="${REFERENCE_CONFIG:-oracle_top2_scattermoe.json}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-oracle_top2_scattermoe_liger.json}"
REFERENCE_BACKEND="${REFERENCE_BACKEND:-scattermoe}"
CANDIDATE_BACKEND="${CANDIDATE_BACKEND:-scattermoe}"
ROOT="$RUN_ROOT/$SLURM_JOB_ID"
LOCAL_ROOT="$(mktemp -d "/tmp/cvcr-liger-parity-${SLURM_JOB_ID}-XXXXXX")"
cleanup() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup EXIT
mkdir -p "$ROOT" /ssd/scxi253/cvcr-research/logs
test "$(sha256sum "$LIGER_WHEEL" | awk '{print $1}')" = "$LIGER_SHA256"
cp -a "$SOURCE/." "$LOCAL_ROOT/source/"
cp -a "$SCATTERMOE_SOURCE/." "$LOCAL_ROOT/scattermoe/"
"$PYTHON" -m pip install --quiet --no-deps --target "$LOCAL_ROOT/liger" "$LIGER_WHEEL"
export PYTHONPATH="$LOCAL_ROOT/source:$LOCAL_ROOT/scattermoe:$LOCAL_ROOT/liger"
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export TRITON_CACHE_DIR="$LOCAL_ROOT/triton"
export OMP_NUM_THREADS=4

srun --ntasks=1 --gres=gpu:1 "$PYTHON" \
  "$LOCAL_ROOT/source/scripts/validate_backend_update.py" \
  --config "$LOCAL_ROOT/source/configs/model/$REFERENCE_CONFIG" \
  --candidate-config "$LOCAL_ROOT/source/configs/model/$CANDIDATE_CONFIG" \
  --data-dir /ssd/scxi253/pretrain500m-20260912-v1/data/fineweb-edu-formal-v1/train-npy \
  --reference-backend "$REFERENCE_BACKEND" \
  --candidate-backend "$CANDIDATE_BACKEND" \
  --reference-microbatch 8 --reference-accumulation 8 \
  --candidate-microbatch "$CANDIDATE_MICROBATCH" \
  --candidate-accumulation "$CANDIDATE_ACCUMULATION" \
  --compile \
  > "$ROOT/parity.json"
sha256sum "$ROOT/parity.json" > "$ROOT/SHA256SUMS"
