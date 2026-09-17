#!/bin/bash
#SBATCH --job-name=cvcr-paired-ce
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/paired-ce-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/paired-ce-%j.err

set -euo pipefail
: "${BASELINE_CHECKPOINT:?set BASELINE_CHECKPOINT}"
: "${CANDIDATE_CHECKPOINT:?set CANDIDATE_CHECKPOINT}"
: "${RUN_NAME:?set RUN_NAME}"
BLOCKS="${BLOCKS:-256}"
BASELINE_CONFIG="${BASELINE_CONFIG:-oracle_top2.json}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-oracle_cvcr_interval4.json}"
SOURCE=/ssd/scxi253/cvcr-research/source
PYTHON=/ssd/scxi253/pretraining2/runtime/venv/bin/python
OUTPUT=/ssd/scxi253/cvcr-research/receipts/${RUN_NAME}-paired-ce.json
LOCAL_ROOT="$(mktemp -d "/tmp/cvcr-paired-ce-${SLURM_JOB_ID}-XXXXXX")"
cleanup() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup EXIT
cp -a "$SOURCE/." "$LOCAL_ROOT/source/"
export PYTHONPATH="$LOCAL_ROOT/source"
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_ROOT/inductor"
export TRITON_CACHE_DIR="$LOCAL_ROOT/triton"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4

"$PYTHON" "$LOCAL_ROOT/source/scripts/paired_block_bootstrap.py" \
  --baseline-config "$LOCAL_ROOT/source/configs/model/$BASELINE_CONFIG" \
  --baseline-checkpoint "$BASELINE_CHECKPOINT" \
  --candidate-config "$LOCAL_ROOT/source/configs/model/$CANDIDATE_CONFIG" \
  --candidate-checkpoint "$CANDIDATE_CHECKPOINT" \
  --data-dir /ssd/scxi253/pretrain500m-20260912-v1/data/fineweb-edu-formal-v1/val-npy \
  --output "$OUTPUT" \
  --blocks "$BLOCKS" \
  --microbatch 2 \
  --bootstrap-repetitions 10000
