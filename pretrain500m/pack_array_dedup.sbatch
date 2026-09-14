#!/usr/bin/env bash
#SBATCH --job-name=p500m-pack14-v2
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=24
#SBATCH --time=10:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-13
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/pack14-v2-%A-%a.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/pack14-v2-%A-%a.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
exclude_root=${P500M_EXCLUDE_ROOT:?P500M_EXCLUDE_ROOT is required}
workers=${P500M_WORKERS:-20}
export PYTHONPATH="$root/overlay:$root/source/pack-v3"
export PYTHONDONTWRITEBYTECODE=1
export P500M_SOURCE_FILE_INDEX="$SLURM_ARRAY_TASK_ID"
part=$(printf 'part-%02d' "$SLURM_ARRAY_TASK_ID")
cd "$root/source/pack-v3"
date -u
hostname
"$py" pack_data.py --source web \
  --tokenizer-dir tokenizer \
  --output-dir "$root/data/fineweb-edu-parts-dedup-v2/$part" \
  --dedup-db "$root/data/fineweb-edu-parts-dedup-v2/$part/dedup.sqlite" \
  --decontam-db "$root/data/decontam-13word.sqlite" \
  --exclude-hash-db "$exclude_root/exclude-$part.sqlite" \
  --target-tokens 2000000000 --validation-tokens 100000000 \
  --shard-tokens 268566528 --workers "$workers" --batch-docs 2048 \
  --progress-every 10000 --no-delete-completed-raw \
  --allow-source-exhaustion --partition-label "$part"
date -u
