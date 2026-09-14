#!/usr/bin/env bash
#SBATCH --job-name=p529m-cpt-lr-v1
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/cpt-lr-v1-%A-%a.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/cpt-lr-v1-%A-%a.err
set -euo pipefail

root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
base="$root/formal/run-v5/resume.pt"
base_sha=13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf
protocol="$root/source/posttrain-v1/cpt-pilot-protocol-v1.json"
expected_protocol_sha=17c91660cfbbd48a3216243be9db9c5e8716c082dc8d8fcf02c75c582e84f993

test "$(sha256sum "$base" | awk '{print $1}')" = "$base_sha"
test "$(sha256sum "$protocol" | awk '{print $1}')" = "$expected_protocol_sha"

lrs=(0.00003 0.0001 0.0003)
lr=${lrs[$SLURM_ARRAY_TASK_ID]}
out="$root/posttrain/cpt-lr-v1-$SLURM_ARRAY_JOB_ID/lr-$lr"
data="$root/data/fineweb-edu-formal-v1"
mkdir -p "$out"

export root py
export PYTHONPATH="$root/source/train-v2"
export P529M_MODEL_SOURCE="$root/source/train-v2"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MASTER_ADDR="$(hostname)"
export MASTER_PORT="$((26000 + (SLURM_ARRAY_JOB_ID + SLURM_ARRAY_TASK_ID) % 14000))"

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:4 \
  bash -lc 'export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-cpt-lr-v1-'"$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"'-inductor"; export TRITON_CACHE_DIR="/tmp/p529m-cpt-lr-v1-'"$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"'-triton"; "$py" -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" "$root/source/posttrain-v1/train_cpt.py" --data-dir '"$data/train-npy"' --val-dir '"$data/val-npy"' --output-dir '"$out"' --architecture deep --target-tokens 536870912 --max-data-epochs 4.0 --microbatch 4 --accumulation 64 --peak-lr '"$lr"' --warmup-tokens 16777216 --save-every 999 --validate-every 64 --dense-bf16-tflops-per-gpu 209.5 --min-mfu 0.50 --mfu-grace-steps 5 --mfu-window 10 --seed 20260914 --deterministic --validate-before-training --init-model-checkpoint '"$base"' --expected-init-sha256 '"$base_sha"' --protocol-file '"$protocol"' --no-save-final' \
  > "$root/logs/cpt-lr-v1-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID-train.log" 2>&1

printf 'CPT_LR_PILOT_V1_COMPLETED lr=%s output=%s\n' "$lr" "$out"
