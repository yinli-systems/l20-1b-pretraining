#!/bin/bash
set -euo pipefail

SOURCE=${SOURCE:?set SOURCE to immutable acquisition source}
PYTHON=${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-/ssd/scxi253/vision1b-research/data/openimages-cvdf-train-1p7m-v1}
LIMIT=${LIMIT:-0}
WORKERS=${WORKERS:-2}
SEGMENT_WORKERS=${SEGMENT_WORKERS:-8}

[[ "$(hostname -s)" == "ln01" ]]
(cd "$SOURCE" && sha256sum --check --quiet SOURCE_SHA256SUMS)
mkdir -p "$OUTPUT_ROOT" /ssd/scxi253/vision1b-research/logs
export PYTHONUNBUFFERED=1
exec "$PYTHON" "$SOURCE/acquire_openimages_cvdf_1p7m.py" \
  --protocol "$SOURCE/openimages_cvdf_1p7m_protocol_v1.json" \
  --output-root "$OUTPUT_ROOT" \
  --workers "$WORKERS" \
  --segment-workers "$SEGMENT_WORKERS" \
  --limit "$LIMIT"
