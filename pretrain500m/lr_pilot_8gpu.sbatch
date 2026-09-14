#!/usr/bin/env bash
#SBATCH --job-name=p529m-lr-v1
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/lr-%A-%a.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/lr-%A-%a.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
export root py
export PYTHONPATH="$root/source/train-v1"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="$((26000 + (SLURM_ARRAY_JOB_ID + SLURM_ARRAY_TASK_ID) % 14000))"
lrs=(0.0015 0.0025 0.0035);lr=${lrs[$SLURM_ARRAY_TASK_ID]}
peak=209.5
if [[ $SLURM_JOB_PARTITION == gpu_4090 ]]; then peak=165.2; fi
name="lr-$lr"
data="$root/data/fineweb-edu-decontam/web/train-npy"
out="$root/pilots/lr-$SLURM_ARRAY_JOB_ID/$name"
mkdir -p "$out"
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:4 \
  bash -lc 'export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-lr-'"$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"'-${SLURM_NODEID}-inductor"; export TRITON_CACHE_DIR="/tmp/p529m-lr-'"$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"'-${SLURM_NODEID}-triton"; "$py" -m torch.distributed.run --nnodes=2 --nproc_per_node=4 --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" train.py --data-dir '"$data"' --val-dir '"$data"' --output-dir '"$out"' --architecture deep --target-tokens 1000000000 --max-data-epochs 100 --microbatch 4 --accumulation 32 --peak-lr '"$lr"' --warmup-tokens 8388608 --save-every 999 --validate-every 30 --dense-bf16-tflops-per-gpu '"$peak"' --min-mfu 0.50 --mfu-grace-steps 2 --mfu-window 4 --pilot-steps 30 --no-save-final --deterministic' \
  > "$root/logs/lr-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID-train.log" 2>&1
printf 'LR_PILOT_COMPLETED lr=%s\n' "$lr"
