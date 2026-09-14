#!/usr/bin/env bash
#SBATCH --job-name=p529m-formal-v1
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=23:30:00
#SBATCH --signal=TERM@300
#SBATCH --requeue
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/formal-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/formal-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
peak_lr=${P500M_PEAK_LR:?P500M_PEAK_LR must be the qualified learning rate}
data="$root/data/fineweb-edu-formal-v1"
out="$root/formal/run-v1"
test -f "$data/manifest.json"
mkdir -p "$out"
export root py
export PYTHONPATH="$root/source/train-v2"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="$((26000 + SLURM_JOB_ID % 14000))"
resume=()
if [[ -f "$out/resume.pt" ]]; then resume=(--resume); fi
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:4 \
  bash -lc 'export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-formal-v1-${SLURM_NODEID}-inductor"; export TRITON_CACHE_DIR="/tmp/p529m-formal-v1-${SLURM_NODEID}-triton"; "$py" -m torch.distributed.run --nnodes=2 --nproc_per_node=4 --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" train.py --data-dir '"$data/train-npy"' --val-dir '"$data/val-npy"' --output-dir '"$out"' --architecture deep --target-tokens 16000000000 --max-data-epochs 1.778555 --microbatch 4 --accumulation 32 --peak-lr '"$peak_lr"' --warmup-tokens 104857600 --save-every 500 --validate-every 250 --dense-bf16-tflops-per-gpu 209.5 --min-mfu 0.50 --mfu-grace-steps 5 --mfu-window 10 --deterministic --protocol-file '"$root/source/protocol-v1.json"' '"${resume[*]}" \
  > "$root/logs/formal-$SLURM_JOB_ID-train.log" 2>&1
printf 'FORMAL_TRAINING_PROCESS_EXITED job=%s lr=%s\n' "$SLURM_JOB_ID" "$peak_lr"
