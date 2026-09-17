#!/bin/bash
#SBATCH --job-name=moe7b-stage-liger-gate
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/ssd/scxi253/cvcr-research/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/cvcr-research/logs/%x-%j.err

set -euo pipefail

SOURCE=/ssd/scxi253/cvcr-research/source-opt-20260917-v43
PYTHON=/ssd/scxi253/pretraining2/runtime/venv/bin/python
LIGER_WHEEL=/ssd/scxi253/l20-1b-posttrain-20260913-v1/wheels/liger-kernel-0.8.2/liger_kernel-0.8.2-py3-none-any.whl
LIGER_SHA256=84c0a7bc9bf4d4cf8ea5ba89ff84d28686afc94215b220851d9f57dc87852741
LOCAL_ROOT=$(mktemp -d "/tmp/moe7b-stage-liger-${SLURM_JOB_ID}-XXXXXX")

cleanup() {
  chmod -R u+w "$LOCAL_ROOT" 2>/dev/null || true
  rm -rf -- "$LOCAL_ROOT"
}
trap cleanup EXIT

test "$(sha256sum "$LIGER_WHEEL" | awk '{print $1}')" = "$LIGER_SHA256"
(cd "$SOURCE" && sha256sum --check --quiet SOURCE_SHA256SUMS)
cp -a "$SOURCE/." "$LOCAL_ROOT/source/"
"$PYTHON" -m pip install --quiet --no-deps --target "$LOCAL_ROOT/liger" "$LIGER_WHEEL"
export PYTHONPATH="$LOCAL_ROOT/source:$LOCAL_ROOT/liger"
export PYTHONPYCACHEPREFIX="$LOCAL_ROOT/pycache"
export CUDA_VISIBLE_DEVICES=0
cd "$LOCAL_ROOT/source"
"$PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_cvcr.py::test_liger_partial_and_all_ignored_targets_are_finite_on_cuda
