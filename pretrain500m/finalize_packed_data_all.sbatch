#!/usr/bin/env bash
#SBATCH --job-name=p500m-finalize-data-all-v1
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/finalize-data-all-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/finalize-data-all-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
parts=${P500M_PARTS_ROOT:?P500M_PARTS_ROOT is required}
receipt=${P500M_DEDUP_RECEIPT:?P500M_DEDUP_RECEIPT is required}
"$py" "$root/source/pack-v3/finalize_packed_data.py" \
  --parts-root "$parts" --dedup-receipt "$receipt" \
  --output-dir "$root/data/fineweb-edu-formal-v1" \
  --expected-parts 14 --use-all-train-tokens
