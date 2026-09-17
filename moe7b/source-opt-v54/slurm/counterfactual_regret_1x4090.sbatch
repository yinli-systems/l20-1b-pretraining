#!/bin/bash
#SBATCH --job-name=cvcr-route-regret
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/regret-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/regret-%j.err

set -euo pipefail
: "${CHECKPOINT:?set CHECKPOINT to a CVCR resume.pt}"
: "${RUN_NAME:?set RUN_NAME}"
SOURCE=/ssd/scxi253/cvcr-research/source
PYTHON=/ssd/scxi253/pretraining2/runtime/venv/bin/python
OUTPUT=/ssd/scxi253/cvcr-research/receipts/${RUN_NAME}-route-regret.json
LOCAL_ROOT="$(mktemp -d "/tmp/cvcr-regret-${SLURM_JOB_ID}-XXXXXX")"
cleanup() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup EXIT
cp -a "$SOURCE/." "$LOCAL_ROOT/source/"
export PYTHONPATH="$LOCAL_ROOT/source"
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_ROOT/inductor"
export TRITON_CACHE_DIR="$LOCAL_ROOT/triton"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4

"$PYTHON" "$LOCAL_ROOT/source/scripts/counterfactual_regret.py" \
  --config "$LOCAL_ROOT/source/configs/model/oracle_cvcr_interval4.json" \
  --checkpoint "$CHECKPOINT" \
  --data-dir /ssd/scxi253/pretrain500m-20260912-v1/data/fineweb-edu-formal-v1/val-npy \
  --output "$OUTPUT" \
  --layers 1,3,5,7 \
  --tokens-per-layer 8 \
  --alternative-candidates 2
