#!/bin/bash
#SBATCH --job-name=cvcr-profile
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/profile-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/profile-%j.err

set -euo pipefail
SOURCE=/ssd/scxi253/cvcr-research/source
PYTHON=/ssd/scxi253/pretraining2/runtime/venv/bin/python
LOCAL_ROOT="$(mktemp -d "/tmp/cvcr-profile-${SLURM_JOB_ID}-XXXXXX")"
cleanup() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup EXIT
cp -a "$SOURCE/." "$LOCAL_ROOT/source/"
export PYTHONPATH="$LOCAL_ROOT/source"
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_ROOT/inductor"
export TRITON_CACHE_DIR="$LOCAL_ROOT/triton"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4

"$PYTHON" "$LOCAL_ROOT/source/scripts/profile_methods.py" \
  "$LOCAL_ROOT/source/configs/model/proxy_top2.json" \
  "$LOCAL_ROOT/source/configs/model/proxy_cvcr.json" \
  --microbatch 2 --warmup 3 --steps 5 --profile
