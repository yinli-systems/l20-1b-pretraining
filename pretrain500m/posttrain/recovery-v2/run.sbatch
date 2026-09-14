#!/usr/bin/env bash
#SBATCH --job-name=p529m-gpu-recovery-v2
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:45:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/recovery-v2-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/recovery-v2-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
src="$root/source/recovery-v2"
inputs="$root/posttrain/recovery-v1-inputs"
out="$root/posttrain/recovery-v2-$SLURM_JOB_ID"
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
export P529M_MODEL_SOURCE="$root/source/train-v2"
export PYTHONPATH="$src:$P529M_MODEL_SOURCE"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-recovery-v2-$SLURM_JOB_ID-inductor"
export TRITON_CACHE_DIR="/tmp/p529m-recovery-v2-$SLURM_JOB_ID-triton"
mkdir -p "$out"
cd "$src"
sha256sum -c SHA256SUMS
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv > "$out/gpu.csv"
common=(--mixture-manifest "$inputs/train.json" --expected-mixture-sha256 "$(sha256sum "$inputs/train.json" | awk '{print $1}')"
  --validation-manifest "$inputs/validation.json" --expected-validation-sha256 "$(sha256sum "$inputs/validation.json" | awk '{print $1}')"
  --admission-receipt "$inputs/engineering-receipt.json" --expected-admission-sha256 "$(sha256sum "$inputs/engineering-receipt.json" | awk '{print $1}')"
  --protocol-file "$inputs/protocol.json" --validation-tokens 131072 --target-tokens 67108864
  --architecture deep --microbatch 4 --accumulation 64 --peak-lr 0.0001 --warmup-tokens 16777216
  --save-every 999 --validate-every 16 --seed 20260914 --deterministic --validate-before-training
  --dense-bf16-tflops-per-gpu 209.5 --min-mfu 0.50 --mfu-grace-steps 5 --mfu-window 10)
initial=(--init-model-checkpoint "$root/formal/run-v5/resume.pt" --expected-init-sha256 13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf)
run_branch() {
  local port=$1
  shift
  srun --nodes=1 --ntasks=1 --cpus-per-task=24 --gres=gpu:4 "$py" -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="$port" "$src/train_cpt.py" "${common[@]}" "$@"
}
port=$((26000 + SLURM_JOB_ID % 10000))
srun --nodes=1 --ntasks=1 --cpus-per-task=24 --gres=gpu:4 "$py" -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="$port" "$src/bucket_probe.py" --output "$out/bucket-probe.json" > "$out/bucket-probe.log" 2>&1
port=$((port+3))
run_branch "$port" --output-dir "$out/continuous" --pilot-steps 32 "${initial[@]}" > "$out/continuous.log" 2>&1
run_branch "$((port+1))" --output-dir "$out/split" --pilot-steps 16 "${initial[@]}" > "$out/split-first.log" 2>&1
run_branch "$((port+2))" --output-dir "$out/split" --pilot-steps 16 --resume > "$out/split-resume.log" 2>&1
"$py" "$src/compare.py" "$out"
