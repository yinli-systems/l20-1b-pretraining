#!/usr/bin/env bash
#SBATCH --job-name=p529m-dispatch-formal-v2
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:20:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/dispatch-formal-v2-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/dispatch-formal-v2-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
"$py" "$root/source/train-v2/dispatch_formal.py"
