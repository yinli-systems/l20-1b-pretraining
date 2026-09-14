#!/usr/bin/env bash
#SBATCH --job-name=p500m-dedup-audit-fast
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/dedup-audit-fast-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/dedup-audit-fast-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
output="$root/data/fineweb-edu-audit-fast-v1"
scratch="$(mktemp -d "/tmp/p500m-dedup-audit-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$output"
trap 'rm -r -- "$scratch"' EXIT
"$py" "$root/source/pack-v3/audit_partitions.py" \
  --parts-root "$root/data/fineweb-edu-parts" \
  --output-dir "$output" \
  --scratch-dir "$scratch" \
  --expected-parts 14
