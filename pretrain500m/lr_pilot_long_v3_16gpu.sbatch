#!/usr/bin/env bash
#SBATCH --job-name=p529m-lr-long-v3
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=01:30:00
#SBATCH --array=0-1
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/lr-long-v3-%A-%a.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/lr-long-v3-%A-%a.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
export root py
export PYTHONPATH="$root/source/train-v2"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="$((26000 + (SLURM_ARRAY_JOB_ID + SLURM_ARRAY_TASK_ID) % 14000))"
lrs=(0.0015 0.0025);lr=${lrs[$SLURM_ARRAY_TASK_ID]}
data="$root/data/fineweb-edu-parts/part-00/web"
out="$root/pilots/lr-long-v3-$SLURM_ARRAY_JOB_ID/lr-$lr"
mkdir -p "$out"
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:4 \
  bash -lc 'export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-lr-long-v3-'"$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"'-${SLURM_NODEID}-inductor"; export TRITON_CACHE_DIR="/tmp/p529m-lr-long-v3-'"$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"'-${SLURM_NODEID}-triton"; "$py" -m torch.distributed.run --nnodes=4 --nproc_per_node=4 --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" "$root/source/train-v2/train_v3.py" --data-dir '"$data/train-npy"' --val-dir '"$data/val-npy"' --output-dir '"$out"' --architecture deep --target-tokens 16000000000 --max-data-epochs 100 --microbatch 4 --accumulation 16 --peak-lr '"$lr"' --warmup-tokens 104857600 --save-every 999 --validate-every 100 --dense-bf16-tflops-per-gpu 209.5 --min-mfu 0.65 --mfu-grace-steps 5 --mfu-window 10 --pilot-steps 300 --no-save-final --deterministic --ddp-gradient-reduction bf16 --ddp-bucket-cap-mb 200 --protocol-file '"$root/source/protocol-v4.json" \
  > "$root/logs/lr-long-v3-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID-train.log" 2>&1
printf 'LONG_LR_PILOT_V3_COMPLETED lr=%s\n' "$lr"
