#!/bin/bash
#SBATCH --job-name=vision-data-1p7m
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=3-00:00:00
#SBATCH --output=/ssd/scxi253/vision1b-research/logs/vision-data-1p7m-%j.out
#SBATCH --error=/ssd/scxi253/vision1b-research/logs/vision-data-1p7m-%j.err

set -euo pipefail
SOURCE=${SOURCE:?set SOURCE to immutable acquisition source}
PYTHON=${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/ssd/scxi253/vision1b-research/data/openimages-cvdf-train-1p7m-v1}
LIMIT=${LIMIT:-0}
WORKERS=${WORKERS:-4}

(cd "$SOURCE" && sha256sum --check --quiet SOURCE_SHA256SUMS)
mkdir -p "$OUTPUT_ROOT" /ssd/scxi253/vision1b-research/logs
"$PYTHON" "$SOURCE/acquire_openimages_cvdf_1p7m.py" \
  --protocol "$SOURCE/openimages_cvdf_1p7m_protocol_v1.json" \
  --output-root "$OUTPUT_ROOT" \
  --workers "$WORKERS" \
  --limit "$LIMIT"
