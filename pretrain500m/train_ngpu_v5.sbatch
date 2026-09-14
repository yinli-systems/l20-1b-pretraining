#!/usr/bin/env bash
#SBATCH --job-name=p529m-formal-v5
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=23:30:00
#SBATCH --signal=TERM@300
#SBATCH --requeue
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/formal-v5-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/formal-v5-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
peak_lr=${P500M_PEAK_LR:?P500M_PEAK_LR must be the qualified learning rate}
protocol_sha=${P500M_PROTOCOL_SHA256:?P500M_PROTOCOL_SHA256 must freeze the formal protocol}
actual_protocol_sha=$(sha256sum "$root/source/protocol-v5.json" | awk '{print $1}')
if [[ "$actual_protocol_sha" != "$protocol_sha" ]]; then
  echo "protocol-v5 hash mismatch" >&2
  exit 4
fi
world=$((SLURM_NNODES * 4))
if (( 256 % world != 0 )); then echo "world size must divide 256" >&2; exit 2; fi
accumulation=$((256 / world))
data="$root/data/fineweb-edu-formal-v1"
out="$root/formal/run-v5"
test -f "$data/manifest.json"
mkdir -p "$out"
if [[ -f "$out/training-status.json" ]] && grep -q '"status": "COMPLETED"' "$out/training-status.json"; then
  echo "formal training is already complete" >&2; exit 3
fi
export root py
export PYTHONPATH="$root/source/train-v2"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="$((26000 + SLURM_JOB_ID % 14000))"
resume=()
if [[ -f "$out/resume.pt" ]]; then resume=(--resume); fi
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:4 \
  bash -lc 'export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-formal-v5-${SLURM_NODEID}-inductor"; export TRITON_CACHE_DIR="/tmp/p529m-formal-v5-${SLURM_NODEID}-triton"; "$py" -m torch.distributed.run --nnodes='"$SLURM_NNODES"' --nproc_per_node=4 --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" "$root/source/train-v2/train.py" --data-dir '"$data/train-npy"' --val-dir '"$data/val-npy"' --output-dir '"$out"' --architecture deep --target-tokens 16000000000 --max-data-epochs 2.1 --microbatch 4 --accumulation '"$accumulation"' --peak-lr '"$peak_lr"' --warmup-tokens 104857600 --save-every 500 --validate-every 250 --dense-bf16-tflops-per-gpu 209.5 --min-mfu 0.50 --mfu-grace-steps 5 --mfu-window 10 --deterministic --protocol-file '"$root/source/protocol-v5.json"' '"${resume[*]}" \
  > "$root/logs/formal-v5-$SLURM_JOB_ID-train.log" 2>&1
printf 'FORMAL_V5_TRAINING_PROCESS_EXITED job=%s lr=%s nodes=%s gpus=%s protocol_sha=%s\n' \
  "$SLURM_JOB_ID" "$peak_lr" "$SLURM_NNODES" "$world" "$protocol_sha"
