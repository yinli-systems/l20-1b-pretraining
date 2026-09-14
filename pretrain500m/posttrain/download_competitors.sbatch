#!/usr/bin/env bash
#SBATCH --job-name=p529m-baseline-download
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/baseline-download-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/baseline-download-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
export HF_HOME="$root/artifacts/baseline-hf-home"
export HF_HUB_DISABLE_TELEMETRY=1
"$py" "$root/source/train-v2/posttrain/download_competitors.py"
