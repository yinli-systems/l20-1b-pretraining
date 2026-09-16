#!/usr/bin/env bash
#SBATCH --job-name=f2cpt529-confirm
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.err
set -euo pipefail

root=/ssd/scxi253/pretrain500m-20260912-v1
src="$root/source/f2-continuation-confirmation-eval-v1"
plan="$src/plan.json"
out="$root/posttrain/f2-continuation-confirmation-eval-v1/$SLURM_JOB_ID"
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
test ! -e "$out"
test -f "$plan"
cd "$src"
sha256sum -c SHA256SUMS
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA GeForce RTX 5090$')" -eq 4
mkdir -p "$out"
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv > "$out/gpu.csv"
export P529M_MODEL_SOURCE="$root/source/train-v2"
export PYTHONPATH="$src:$P529M_MODEL_SOURCE"
export OMP_NUM_THREADS=4
port=$((29000 + SLURM_JOB_ID % 10000))
srun --nodes=1 --ntasks=1 --cpus-per-task=24 --gres=gpu:4 \
  "$py" -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port="$port" "$src/evaluate.py" \
  --plan "$plan" --expected-plan-sha256 919d5738976491dd3d614a104372e5e7b7a57d21636468d7dad68acc8eba5310 \
  --microbatch 4 --output "$out/result.json"

sha256sum "$out/result.json"
