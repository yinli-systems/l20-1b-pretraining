#!/bin/bash
#SBATCH --job-name=cvcr-backend-parity
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/backend-parity-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/backend-parity-%j.err

set -euo pipefail
SOURCE="${SOURCE:?set SOURCE to the immutable candidate source directory}"
SCATTERMOE_SOURCE="${SCATTERMOE_SOURCE:?set SCATTERMOE_SOURCE to the pinned checkout}"
PYTHON="${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}"
CANDIDATE_BACKEND="${CANDIDATE_BACKEND:-scattermoe}"
RUN_ROOT="${RUN_ROOT:-/ssd/scxi253/cvcr-research/backend-update-parity}"
RUN="$RUN_ROOT/$SLURM_JOB_ID"
LOCAL_ROOT="$(mktemp -d "/tmp/cvcr-parity-${SLURM_JOB_ID}-XXXXXX")"
cleanup() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup EXIT
mkdir -p "$RUN" /ssd/scxi253/cvcr-research/logs
cp -a "$SOURCE/." "$LOCAL_ROOT/source/"
cp -a "$SCATTERMOE_SOURCE/." "$LOCAL_ROOT/scattermoe/"
export PYTHONPATH="$LOCAL_ROOT/source:$LOCAL_ROOT/scattermoe"
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export TRITON_CACHE_DIR="$LOCAL_ROOT/triton"
export OMP_NUM_THREADS=6

"$PYTHON" "$LOCAL_ROOT/source/scripts/validate_backend_update.py" \
  --config "$LOCAL_ROOT/source/configs/model/oracle_top2.json" \
  --data-dir /ssd/scxi253/pretrain500m-20260912-v1/data/fineweb-edu-formal-v1/train-npy \
  --microbatch 2 \
  --accumulation 2 \
  --candidate-backend "$CANDIDATE_BACKEND" \
  | tee "$RUN/parity.json"
sha256sum "$RUN/parity.json" > "$RUN/SHA256SUMS"
