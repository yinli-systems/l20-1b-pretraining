#!/usr/bin/env bash
#SBATCH --job-name=p530m-batch-v1
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/batch-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/batch-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
export PYTHONPATH="$root/source/shape-v1"
export OMP_NUM_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="/tmp/p530m-$SLURM_JOB_ID-inductor"
export TRITON_CACHE_DIR="/tmp/p530m-$SLURM_JOB_ID-triton"
out="$root/receipts/batch-$SLURM_JOB_ID";mkdir "$out";cd "$root/source/shape-v1"
for mb in 3 4 5; do
  "$py" -m torch.distributed.run --standalone --nproc_per_node=4 gpu_probe.py --compile --deep --microbatch "$mb" --output "$out/deep-mb$mb.json" > "$out/deep-mb$mb.log" 2>&1 || { test "$mb" = 5 || exit 1; }
done
"$py" -m torch.distributed.run --standalone --nproc_per_node=4 gpu_probe.py --compile --microbatch 6 --output "$out/wide-mb6.json" > "$out/wide-mb6.log" 2>&1 || true
printf 'BATCH_QUALIFICATION_COMPLETED\n'
