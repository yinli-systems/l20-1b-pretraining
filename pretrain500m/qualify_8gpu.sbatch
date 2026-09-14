#!/usr/bin/env bash
#SBATCH --job-name=p530m-8gpu-v1
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/8gpu-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/8gpu-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
export root py
export PYTHONPATH="$root/source/shape-v1"
export OMP_NUM_THREADS=4
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
out="$root/receipts/8gpu-$SLURM_JOB_ID";mkdir "$out";cd "$root/source/shape-v1"
run_probe() {
  local name=$1; shift
  srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:4 \
    bash -lc 'export TORCHINDUCTOR_CACHE_DIR="/tmp/p530m-'"$SLURM_JOB_ID"'-${SLURM_NODEID}-inductor"; export TRITON_CACHE_DIR="/tmp/p530m-'"$SLURM_JOB_ID"'-${SLURM_NODEID}-triton"; "$py" -m torch.distributed.run --nnodes=2 --nproc_per_node=4 --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" gpu_probe.py '"$*" \
    > "$out/$name.log" 2>&1
}
run_probe wide-mb4-acc32 --compile --microbatch 4 --accumulation 32 --steps 6 --warmup 2 --output "$out/wide-mb4-acc32.json"
run_probe deep-mb4-acc32 --compile --deep --microbatch 4 --accumulation 32 --steps 6 --warmup 2 --output "$out/deep-mb4-acc32.json"
printf 'EIGHT_GPU_QUALIFICATION_COMPLETED\n'
