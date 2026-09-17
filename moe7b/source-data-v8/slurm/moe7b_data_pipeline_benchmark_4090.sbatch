#!/bin/bash
#SBATCH --job-name=moe7b-data-pipe-bench
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --output=/ssd/scxi253/cvcr-research/data/moe7b-150b-v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/data/moe7b-150b-v1/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT=/ssd/scxi253/cvcr-research/source-data-20260917-v8
DATA_ROOT=/ssd/scxi253/cvcr-research/data/moe7b-150b-v1
RAW_ROOT=/data/run01/scxi253/moe7b-150b-v1/raw-production-v1
RUNTIME=/ssd/scxi253/pretraining2/runtime/venv
TOKENIZER=/ssd/scxi253/pretrain500m-20260912-v1/source/pack-v3/tokenizer
CONTAMINATION_INDEX=${DATA_ROOT}/control/public-eval-contamination-index-v2.json.gz
BENCH_SECONDS=${BENCH_SECONDS:-240}
BATCH_DOCUMENTS=${BATCH_DOCUMENTS:-64}
PIPELINE_MODE=${PIPELINE_MODE:-analysis}
case "${PIPELINE_MODE}" in
  analysis)
    pipeline_args=(--pipeline-analysis)
    WORKERS=${WORKERS:-$((SLURM_CPUS_PER_TASK / 2))}
    RAYON_THREADS=${RAYON_THREADS:-$((SLURM_CPUS_PER_TASK - WORKERS))}
    TOKENIZER_PARALLELISM=true
    ;;
  tokenization)
    pipeline_args=(--pipeline-tokenization)
    WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK}}
    RAYON_THREADS=1
    TOKENIZER_PARALLELISM=false
    ;;
  serial)
    pipeline_args=()
    WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK}}
    RAYON_THREADS=${RAYON_THREADS:-${SLURM_CPUS_PER_TASK}}
    TOKENIZER_PARALLELISM=true
    ;;
  *) echo "unsupported PIPELINE_MODE=${PIPELINE_MODE}" >&2; exit 64 ;;
esac
if (( WORKERS < 1 || RAYON_THREADS < 1 )); then
  echo "WORKERS and RAYON_THREADS must be positive" >&2
  exit 64
fi
BENCH_ROOT=${DATA_ROOT}/bench-pipeline-v8/job-${SLURM_JOB_ID}-${PIPELINE_MODE}-cpu${SLURM_CPUS_PER_TASK}-w${WORKERS}-r${RAYON_THREADS}-batch${BATCH_DOCUMENTS}

mkdir -p "${DATA_ROOT}/logs" "${BENCH_ROOT}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=${TOKENIZER_PARALLELISM}
export RAYON_NUM_THREADS=${RAYON_THREADS}
export OMP_NUM_THREADS=1

started=$(date +%s)
set +e
"${RUNTIME}/bin/python" "${PROJECT_ROOT}/scripts/build_moe7b_full_data.py" \
  --build "${PROJECT_ROOT}/data/moe7b_150b_build_v1.json" \
  --output-root "${BENCH_ROOT}" \
  --raw-root "${RAW_ROOT}" \
  --tokenizer-dir "${TOKENIZER}" \
  --contamination-index "${CONTAMINATION_INDEX}" \
  --offline --source-wait-seconds 60 \
  --component pes2o \
  --workers "${WORKERS}" \
  --batch-documents "${BATCH_DOCUMENTS}" \
  "${pipeline_args[@]}" &
builder_pid=$!
sleep "${BENCH_SECONDS}"
kill -TERM "${builder_pid}" 2>/dev/null
for _ in $(seq 1 30); do
  if ! kill -0 "${builder_pid}" 2>/dev/null; then
    break
  fi
  sleep 1
done
if kill -0 "${builder_pid}" 2>/dev/null; then
  kill -KILL "${builder_pid}" 2>/dev/null
fi
wait "${builder_pid}"
builder_rc=$?
set -e
finished=$(date +%s)
train_bytes=$(find "${BENCH_ROOT}/packed/pes2o" -maxdepth 1 -type f -name '*.train.bin.tmp-*' -printf '%s\n' 2>/dev/null | awk '{sum += $1} END {print sum + 0}')
dedup_bytes=$(find "${BENCH_ROOT}/packed/pes2o" -maxdepth 1 -type f -name '*.dedup.bin.tmp-*' -printf '%s\n' 2>/dev/null | awk '{sum += $1} END {print sum + 0}')
echo "BENCH_RESULT job=${SLURM_JOB_ID} pipeline_mode=${PIPELINE_MODE} cpus=${SLURM_CPUS_PER_TASK} workers=${WORKERS} rayon_threads=${RAYON_THREADS} batch_documents=${BATCH_DOCUMENTS} elapsed_seconds=$((finished-started)) train_bytes=${train_bytes} approximate_token_ids=$((train_bytes/2)) dedup_bytes=${dedup_bytes} builder_rc=${builder_rc}"
