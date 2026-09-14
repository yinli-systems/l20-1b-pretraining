#!/usr/bin/env bash
#SBATCH --job-name=p529m-resume-det-v2
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:40:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/resume-det-v2-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/resume-det-v2-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
export root py
export PYTHONPATH="$root/source/train-v2"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="$((28000 + SLURM_JOB_ID % 12000))"
data="$root/data/fineweb-edu-decontam/web/train-npy"
base=(--data-dir "$data" --val-dir "$data" --architecture deep --target-tokens 100000000
      --microbatch 4 --accumulation 32 --peak-lr 0.0015 --warmup-tokens 104857600
      --save-every 2 --validate-every 4 --dense-bf16-tflops-per-gpu 209.5
      --min-mfu 0.65 --mfu-grace-steps 1 --mfu-window 2 --deterministic
      --protocol-file "$root/source/protocol-v2.json")
run_train() {
  local name=$1; shift
  srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:4 \
    bash -lc 'export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-det-v2-'"$SLURM_JOB_ID"'-${SLURM_NODEID}-inductor"; export TRITON_CACHE_DIR="/tmp/p529m-det-v2-'"$SLURM_JOB_ID"'-${SLURM_NODEID}-triton"; "$py" -m torch.distributed.run --nnodes=2 --nproc_per_node=4 --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" "$root/source/train-v2/train.py" '"$*" \
    > "$root/logs/$name-$SLURM_JOB_ID.log" 2>&1
}
run_train det-v2-continuous "${base[@]}" --output-dir "$root/pilots/$SLURM_JOB_ID/continuous" --pilot-steps 4
run_train det-v2-split-a "${base[@]}" --output-dir "$root/pilots/$SLURM_JOB_ID/split" --pilot-steps 2
run_train det-v2-split-b "${base[@]}" --output-dir "$root/pilots/$SLURM_JOB_ID/split" --pilot-steps 2 --resume
"$py" "$root/source/train-v2/compare_checkpoints.py" \
  "$root/pilots/$SLURM_JOB_ID/continuous/resume.pt" "$root/pilots/$SLURM_JOB_ID/split/resume.pt" \
  | tee "$root/pilots/$SLURM_JOB_ID/resume-comparison.json"
printf 'DETERMINISTIC_RESUME_V2_PILOT_COMPLETED\n'
