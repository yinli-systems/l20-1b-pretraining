#!/usr/bin/env bash
#SBATCH --job-name=p500m-pack-v1
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=7
#SBATCH --cpus-per-task=42
#SBATCH --time=10:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/pack-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/pack-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
export PYTHONPATH="$root/overlay:$root/source/pack-v1"
export PYTHONDONTWRITEBYTECODE=1
cd "$root/source/pack-v1"
date -u
hostname
"$py" pack_data.py --source web \
  --tokenizer-dir tokenizer \
  --output-dir "$root/data/fineweb-edu-decontam" \
  --dedup-db "$root/data/fineweb-edu-dedup.sqlite" \
  --decontam-db "$root/data/decontam-13word.sqlite" \
  --target-tokens 9000000000 --validation-tokens 5000000 \
  --shard-tokens 268566528 --workers 36 --batch-docs 2048 \
  --progress-every 10000 --no-delete-completed-raw
date -u
