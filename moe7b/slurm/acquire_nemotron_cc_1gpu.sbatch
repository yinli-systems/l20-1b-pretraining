#!/bin/bash
#SBATCH --job-name=moe-data-nemotron
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=7-00:00:00
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/moe-data-nemotron-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/moe-data-nemotron-%j.err

set -euo pipefail
SOURCE=${SOURCE:?set SOURCE to immutable acquisition source}
PYTHON=${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/scxi253/moe7b-data-sources/nemotron-cc-v1}
LIMIT=${LIMIT:-0}
WORKERS=${WORKERS:-6}

(cd "$SOURCE" && sha256sum --check --quiet SOURCE_SHA256SUMS)
mkdir -p "$OUTPUT_ROOT" /ssd/scxi253/cvcr-research/logs
"$PYTHON" "$SOURCE/acquire_nemotron_cc.py" \
  --protocol "$SOURCE/nemotron_cc_acquisition_protocol_v1.json" \
  --output-root "$OUTPUT_ROOT" \
  --workers "$WORKERS" \
  --limit "$LIMIT"
