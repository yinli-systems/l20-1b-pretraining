#!/usr/bin/env bash
#SBATCH --job-name=p529m-pilot-v1
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/pilot-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/pilot-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
export root py
export PYTHONPATH="$root/source/train-v1"
export OMP_NUM_THREADS=4
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="$((24000 + SLURM_JOB_ID % 16000))"
export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-$SLURM_JOB_ID-${SLURM_NODEID:-batch}-inductor"
export TRITON_CACHE_DIR="/tmp/p529m-$SLURM_JOB_ID-${SLURM_NODEID:-batch}-triton"
data="$root/data/fineweb-edu-decontam/web/train-npy"
base=(--data-dir "$data" --val-dir "$data" --architecture deep --target-tokens 100000000
      --microbatch 4 --accumulation 32 --peak-lr 0.0006 --warmup-tokens 104857600
      --save-every 6 --validate-every 6 --dense-bf16-tflops-per-gpu 209.5
      --min-mfu 0.50 --mfu-grace-steps 2 --mfu-window 4)
run_train() {
  local name=$1; shift
  srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:4 \
    bash -lc 'export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-'"$SLURM_JOB_ID"'-${SLURM_NODEID}-inductor"; export TRITON_CACHE_DIR="/tmp/p529m-'"$SLURM_JOB_ID"'-${SLURM_NODEID}-triton"; "$py" -m torch.distributed.run --nnodes=2 --nproc_per_node=4 --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" train.py '"$*" \
    > "$root/logs/$name-$SLURM_JOB_ID.log" 2>&1
}
run_train pilot-continuous "${base[@]}" --output-dir "$root/pilots/$SLURM_JOB_ID/continuous" --pilot-steps 12
run_train pilot-split-a "${base[@]}" --output-dir "$root/pilots/$SLURM_JOB_ID/split" --pilot-steps 6
run_train pilot-split-b "${base[@]}" --output-dir "$root/pilots/$SLURM_JOB_ID/split" --pilot-steps 6 --resume
"$py" "$root/source/train-v1/compare_checkpoints.py" \
  "$root/pilots/$SLURM_JOB_ID/continuous/resume.pt" "$root/pilots/$SLURM_JOB_ID/split/resume.pt" \
  | tee "$root/pilots/$SLURM_JOB_ID/resume-comparison.json"
printf 'REAL_DATA_PILOT_COMPLETED\n'
