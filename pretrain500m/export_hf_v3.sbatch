#!/usr/bin/env bash
#SBATCH --job-name=p529m-export-v3
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/export-v3-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/export-v3-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
export PYTHONPATH="$root/source/train-v2"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$py" "$root/source/train-v2/export_hf.py" \
  --checkpoint "$root/formal/run-v3/resume.pt" \
  --tokenizer-dir "$root/source/pack-v3/tokenizer" \
  --output-dir "$root/formal/hf-v3" --expected-step 7629
