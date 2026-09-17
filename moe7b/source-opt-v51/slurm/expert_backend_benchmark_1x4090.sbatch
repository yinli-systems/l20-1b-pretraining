#!/bin/bash
#SBATCH --job-name=cvcr-expert-backends
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/expert-backends-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/expert-backends-%j.err

set -euo pipefail
SOURCE="${SOURCE:?set SOURCE to the immutable candidate source directory}"
SCATTERMOE_SOURCE="${SCATTERMOE_SOURCE:?set SCATTERMOE_SOURCE to the pinned checkout}"
PYTHON="${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-/ssd/scxi253/cvcr-research/backend-benchmarks}"
RUN="$RUN_ROOT/${SLURM_JOB_ID}"
LOCAL_ROOT="$(mktemp -d "/tmp/cvcr-expert-${SLURM_JOB_ID}-XXXXXX")"
cleanup() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup EXIT
mkdir -p "$RUN" /ssd/scxi253/cvcr-research/logs
cp -a "$SOURCE/." "$LOCAL_ROOT/source/"
cp -a "$SCATTERMOE_SOURCE/." "$LOCAL_ROOT/scattermoe/"
export PYTHONPATH="$LOCAL_ROOT/source:$LOCAL_ROOT/scattermoe"
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_ROOT/inductor"
export TRITON_CACHE_DIR="$LOCAL_ROOT/triton"
export OMP_NUM_THREADS=6

"$PYTHON" "$LOCAL_ROOT/source/scripts/benchmark_expert_backends.py" \
  --config "$LOCAL_ROOT/source/configs/model/oracle_top2.json" \
  --tokens 16384 32768 \
  --distributions balanced skewed empty \
  --warmup 10 \
  --steps 50 \
  --trace-dir "$RUN/traces" \
  | tee "$RUN/results.jsonl"

sha256sum "$RUN/results.jsonl" > "$RUN/SHA256SUMS"
find "$RUN/traces" -type f -print0 | sort -z | xargs -0 sha256sum >> "$RUN/SHA256SUMS"
