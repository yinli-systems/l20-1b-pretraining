#!/usr/bin/env bash
#SBATCH --job-name=p500m-dedup-audit-v2
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/dedup-audit-v2-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/dedup-audit-v2-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
scratch="$(mktemp -d "/tmp/p500m-dedup-audit-v2-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$root/data/fineweb-edu-audit-v2"
trap 'rm -r -- "$scratch"' EXIT
"$py" "$root/source/pack-v3/audit_partitions.py" \
  --parts-root "$root/data/fineweb-edu-parts-dedup-v2" \
  --output-dir "$root/data/fineweb-edu-audit-v2" \
  --scratch-dir "$scratch" \
  --expected-parts 14
