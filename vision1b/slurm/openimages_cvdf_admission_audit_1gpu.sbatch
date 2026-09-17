#!/bin/bash
#SBATCH --job-name=vision-data-1p7m-admission-audit
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --output=/ssd/scxi253/vision1b-research/logs/vision-data-1p7m-admission-audit-%j.out
#SBATCH --error=/ssd/scxi253/vision1b-research/logs/vision-data-1p7m-admission-audit-%j.err

set -euo pipefail
SOURCE=${SOURCE:?set SOURCE to immutable admission source}
PYTHON=${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}
DATA=${DATA:-/ssd/scxi253/vision1b-research/data/openimages-cvdf-train-1p7m-v1}
ROOT=${ADMISSION_ROOT:-/data/scxi253/vision1b-data/openimages-cvdf-train-1p7m-admission-v1}
LOCAL_ROOT=$(mktemp -d "/tmp/vision1b-admission-audit-$SLURM_JOB_ID-XXXXXX")
trap 'chmod -R u+w "$LOCAL_ROOT" 2>/dev/null || true; rm -rf -- "$LOCAL_ROOT"' EXIT
export PYTHONPATH="/ssd/scxi253/vision1b-research/deps/pillow${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"

(cd "$SOURCE" && sha256sum --check --quiet SOURCE_SHA256SUMS)
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
mkdir -p "$ROOT/audit" /ssd/scxi253/vision1b-research/logs

"$PYTHON" "$SOURCE/openimages_cvdf_admission.py" render-audit \
  --protocol "$SOURCE/openimages_cvdf_admission_protocol_v1.json" \
  --reduction-receipt "$ROOT/reduced/reduction-receipt.json" \
  --archive-root "$DATA/tar" \
  --output-root "$ROOT/audit"
