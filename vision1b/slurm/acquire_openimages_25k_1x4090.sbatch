#!/bin/bash
#SBATCH --job-name=oi25k-acquire
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:30:00
#SBATCH --output=/ssd/scxi253/vision1b-research/logs/oi25k-acquire-%j.out
#SBATCH --error=/ssd/scxi253/vision1b-research/logs/oi25k-acquire-%j.err

set -euo pipefail
PYTHON=/ssd/scxi253/pretraining2/runtime/venv/bin/python
SOURCE=/ssd/scxi253/vision1b-research/source
ROOT=/ssd/scxi253/vision1b-research/data/openimages-real-25k-v1
METADATA=/ssd/scxi253/vision1b-research/data/openimages-real-pilot-v1/sources/train-images-boxable-with-rotation.csv
export PYTHONPATH=/ssd/scxi253/vision1b-research/deps/pillow${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$ROOT" /ssd/scxi253/vision1b-research/logs
"$PYTHON" "$SOURCE/acquire_openimages_25k.py" \
  --protocol "$SOURCE/real_data_25k_protocol.json" \
  --metadata "$METADATA" \
  --output-root "$ROOT" \
  --workers 32
