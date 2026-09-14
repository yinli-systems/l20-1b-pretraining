#!/usr/bin/env bash
#SBATCH --job-name=p500m-dedup-audit-v1
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/dedup-audit-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/dedup-audit-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
mkdir -p "$root/data/fineweb-edu-audit-v1"
"$py" "$root/source/pack-v3/audit_partitions.py" \
  --parts-root "$root/data/fineweb-edu-parts" \
  --output-dir "$root/data/fineweb-edu-audit-v1" \
  --expected-parts 14
