#!/usr/bin/env bash
#SBATCH --job-name=p530m-shape-v1
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/shape-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/shape-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
export PYTHONPATH="$root/source/shape-v1"
export OMP_NUM_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="/tmp/p530m-$SLURM_JOB_ID-inductor"
export TRITON_CACHE_DIR="/tmp/p530m-$SLURM_JOB_ID-triton"
out="$root/receipts/shape-$SLURM_JOB_ID";mkdir "$out";cd "$root/source/shape-v1"
for mb in 2 3 4; do
  "$py" -m torch.distributed.run --standalone --nproc_per_node=4 gpu_probe.py --compile --microbatch "$mb" --output "$out/wide-mb$mb.json" > "$out/wide-mb$mb.log" 2>&1 || { test "$mb" = 4 || exit 1; }
done
"$py" -m torch.distributed.run --standalone --nproc_per_node=4 gpu_probe.py --compile --deep --microbatch 2 --output "$out/deep-mb2.json" > "$out/deep-mb2.log" 2>&1
printf 'SHAPE_QUALIFICATION_COMPLETED\n'
