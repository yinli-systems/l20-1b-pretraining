#!/bin/bash
#SBATCH --job-name=vision-data-1p7m
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=14-00:00:00
#SBATCH --output=/ssd/scxi253/vision1b-research/logs/vision-data-1p7m-%j.out
#SBATCH --error=/ssd/scxi253/vision1b-research/logs/vision-data-1p7m-%j.err

set -euo pipefail
SOURCE=${SOURCE:?set SOURCE to immutable acquisition source}
PYTHON=${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/ssd/scxi253/vision1b-research/data/openimages-cvdf-train-1p7m-v1}
LIMIT=${LIMIT:-0}
WORKERS=${WORKERS:-2}
SEGMENT_WORKERS=${SEGMENT_WORKERS:-8}
LOGIN_HOST=${LOGIN_HOST:-ln01}

gpu_inventory=$(nvidia-smi --query-gpu=name --format=csv,noheader)
gpu_count=$(printf '%s\n' "$gpu_inventory" | sed '/^$/d' | wc -l | tr -d ' ')
[[ "$gpu_count" == "1" ]]
[[ "$gpu_inventory" == "NVIDIA GeForce RTX 4090" ]]
(cd "$SOURCE" && sha256sum --check --quiet SOURCE_SHA256SUMS)

ssh -n \
  -o BatchMode=yes \
  -o ConnectTimeout=10 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=6 \
  -o StrictHostKeyChecking=yes \
  "$LOGIN_HOST" \
  env \
    SOURCE="$SOURCE" \
    PYTHON="$PYTHON" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    LIMIT="$LIMIT" \
    WORKERS="$WORKERS" \
    SEGMENT_WORKERS="$SEGMENT_WORKERS" \
    "$SOURCE/run_openimages_cvdf_on_login.sh"
