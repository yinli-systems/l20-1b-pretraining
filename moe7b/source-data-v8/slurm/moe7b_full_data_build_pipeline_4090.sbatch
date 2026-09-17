#!/bin/bash
#SBATCH --job-name=moe7b-data-150b-pipe
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=30-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ssd/scxi253/cvcr-research/data/moe7b-150b-v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/data/moe7b-150b-v1/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT=/ssd/scxi253/cvcr-research/source-data-20260917-v8
DATA_ROOT=/ssd/scxi253/cvcr-research/data/moe7b-150b-v1
BUILD_ROOT=${DATA_ROOT}/production-v1
RAW_ROOT=/data/run01/scxi253/moe7b-150b-v1/raw-production-v1
RUNTIME=/ssd/scxi253/pretraining2/runtime/venv
TOKENIZER=/ssd/scxi253/pretrain500m-20260912-v1/source/pack-v3/tokenizer
CONTAMINATION_INDEX=${DATA_ROOT}/control/public-eval-contamination-index-v2.json.gz
BATCH_DOCUMENTS=${BATCH_DOCUMENTS:-64}

mkdir -p "${DATA_ROOT}/logs" "${BUILD_ROOT}/state" "${RAW_ROOT}"
exec 9>"${BUILD_ROOT}/state/writer.lock"
if ! flock -n 9; then
  echo "refusing a second production data writer: writer.lock is held" >&2
  exit 75
fi
if pgrep -af 'build_moe7b_full_data.py' | grep -F -- "--output-root ${BUILD_ROOT}" | grep -v "$$"; then
  echo "refusing a second production data writer: another matching process is live" >&2
  exit 76
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=3
export OMP_NUM_THREADS=1

"${RUNTIME}/bin/python" "${PROJECT_ROOT}/scripts/build_moe7b_full_data.py" \
  --build "${PROJECT_ROOT}/data/moe7b_150b_build_v1.json" \
  --output-root "${BUILD_ROOT}" \
  --raw-root "${RAW_ROOT}" \
  --tokenizer-dir "${TOKENIZER}" \
  --contamination-index "${CONTAMINATION_INDEX}" \
  --offline --source-wait-seconds 86400 \
  --workers 3 \
  --batch-documents "${BATCH_DOCUMENTS}" \
  --pipeline-analysis

"${RUNTIME}/bin/python" "${PROJECT_ROOT}/scripts/finalize_moe7b_stage_manifest.py" \
  --build-root "${BUILD_ROOT}" \
  --build-contract "${PROJECT_ROOT}/data/moe7b_150b_build_v1.json" \
  --output "${BUILD_ROOT}/stage-mixture-manifest-v1.json"
