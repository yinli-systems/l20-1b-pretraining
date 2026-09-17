#!/bin/bash
#SBATCH --job-name=cvcr-oracle
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/oracle-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/oracle-%j.err

set -euo pipefail
: "${CHECKPOINT:?set CHECKPOINT to a CVCR resume.pt}"
: "${RUN_NAME:?set RUN_NAME for the oracle receipt}"
MODEL_CONFIG="${MODEL_CONFIG:-oracle_cvcr.json}"
BATCHES="${BATCHES:-2}"
ORACLE_TOKENS="${ORACLE_TOKENS:-256}"
SOURCE=/ssd/scxi253/cvcr-research/source
PYTHON=/ssd/scxi253/pretraining2/runtime/venv/bin/python
OUTPUT=/ssd/scxi253/cvcr-research/receipts/${RUN_NAME}-oracle.json
LOCAL_ROOT="$(mktemp -d "/tmp/cvcr-oracle-${SLURM_JOB_ID}-XXXXXX")"
cleanup() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup EXIT
cp -a "$SOURCE/." "$LOCAL_ROOT/source/"
export PYTHONPATH="$LOCAL_ROOT/source"
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4

"$PYTHON" "$LOCAL_ROOT/source/scripts/oracle_diagnostics.py" \
  --config "$LOCAL_ROOT/source/configs/model/$MODEL_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --data-dir /ssd/scxi253/pretrain500m-20260912-v1/data/fineweb-edu-formal-v1/val-npy \
  --output "$OUTPUT" \
  --batches "$BATCHES" \
  --oracle-tokens "$ORACLE_TOKENS"
