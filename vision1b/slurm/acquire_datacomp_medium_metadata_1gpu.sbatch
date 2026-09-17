#!/bin/bash
#SBATCH --job-name=vision-data-datacomp-meta
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=2-00:00:00
#SBATCH --output=/ssd/scxi253/vision1b-research/logs/vision-data-datacomp-meta-%j.out
#SBATCH --error=/ssd/scxi253/vision1b-research/logs/vision-data-datacomp-meta-%j.err

set -euo pipefail
SOURCE=${SOURCE:?set SOURCE to immutable acquisition source}
PYTHON=${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/scxi253/vision1b-sources/datacomp-medium-metadata-v1}
LIMIT=${LIMIT:-0}
WORKERS=${WORKERS:-4}

(cd "$SOURCE" && sha256sum --check --quiet SOURCE_SHA256SUMS)
mkdir -p "$OUTPUT_ROOT" /ssd/scxi253/vision1b-research/logs
"$PYTHON" "$SOURCE/acquire_datacomp_medium_metadata.py" \
  --protocol "$SOURCE/datacomp_medium_metadata_protocol_v1.json" \
  --inventory "$SOURCE/datacomp_medium_parquet_inventory_v1.jsonl" \
  --output-root "$OUTPUT_ROOT" \
  --workers "$WORKERS" \
  --limit "$LIMIT"
