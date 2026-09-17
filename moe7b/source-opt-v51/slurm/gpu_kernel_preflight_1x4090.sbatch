#!/bin/bash
#SBATCH --job-name=cvcr-kernel-v1
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/kernel-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/kernel-%j.err

set -euo pipefail
SOURCE=/ssd/scxi253/cvcr-research/source
PYTHON=/ssd/scxi253/pretraining2/runtime/venv/bin/python
RECEIPT=/ssd/scxi253/cvcr-research/receipts/kernel-${SLURM_JOB_ID}.json
LOCAL_ROOT="$(mktemp -d "/tmp/cvcr-kernel-${SLURM_JOB_ID}-XXXXXX")"
cleanup() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup EXIT
cp -a "$SOURCE/." "$LOCAL_ROOT/source/"
export PYTHONPATH="$LOCAL_ROOT/source"
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_ROOT/inductor"
export TRITON_CACHE_DIR="$LOCAL_ROOT/triton"
export CUDA_DEVICE_MAX_CONNECTIONS=1
mkdir -p "$(dirname "$RECEIPT")"
timeout 120s "$PYTHON" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)'
"$PYTHON" "$LOCAL_ROOT/source/scripts/gpu_kernel_preflight.py" \
  --output "$RECEIPT" \
  --measure-steps 10
