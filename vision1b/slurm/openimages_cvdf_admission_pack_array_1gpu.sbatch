#!/bin/bash
#SBATCH --job-name=vision-data-1p7m-admission-pack
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --array=0-15%4
#SBATCH --time=1-00:00:00
#SBATCH --output=/ssd/scxi253/vision1b-research/logs/vision-data-1p7m-admission-pack-%A-%a.out
#SBATCH --error=/ssd/scxi253/vision1b-research/logs/vision-data-1p7m-admission-pack-%A-%a.err

set -euo pipefail
SOURCE=${SOURCE:?set SOURCE to immutable admission source}
PYTHON=${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}
DATA=${DATA:-/ssd/scxi253/vision1b-research/data/openimages-cvdf-train-1p7m-v1}
ROOT=${ADMISSION_ROOT:-/data/run01/scxi253/vision1b-data/openimages-cvdf-train-1p7m-admission-v1}
SHARDS=(0 1 2 3 4 5 6 7 8 9 a b c d e f)
SHARD=${SHARDS[$SLURM_ARRAY_TASK_ID]}
LOCAL_ROOT=$(mktemp -d "/tmp/vision1b-admission-pack-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID-XXXXXX")
trap 'chmod -R u+w "$LOCAL_ROOT" 2>/dev/null || true; rm -rf -- "$LOCAL_ROOT"' EXIT
export PYTHONPATH="/ssd/scxi253/vision1b-research/deps/pillow${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"

(cd "$SOURCE" && sha256sum --check --quiet SOURCE_SHA256SUMS)
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
mkdir -p "$ROOT/packed" /ssd/scxi253/vision1b-research/logs

"$PYTHON" "$SOURCE/openimages_cvdf_admission.py" pack-shard \
  --protocol "$SOURCE/openimages_cvdf_admission_protocol_v1.json" \
  --reduction-receipt "$ROOT/reduced/reduction-receipt.json" \
  --archive "$DATA/tar/train_${SHARD}.tar.gz" \
  --output-tar "$ROOT/packed/train_${SHARD}.tar" \
  --output-receipt "$ROOT/packed/train_${SHARD}.receipt.json"
