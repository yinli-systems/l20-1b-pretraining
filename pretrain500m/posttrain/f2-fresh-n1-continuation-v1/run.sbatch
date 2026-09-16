#!/usr/bin/env bash
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.err
set -euo pipefail

root=/ssd/scxi253/pretrain500m-20260912-v1
bundle="$root/source/f2-fresh-n1-continuation-v1"
src="$root/source/cpt-confirmation-v1"
inputs="$root/source/new-pool-mixture-pilot-inputs-v1"
arm=${ARM_ID:?ARM_ID must name the frozen arm}
peak_lr=${PEAK_LR:?PEAK_LR must name the frozen learning rate}
seed=20260915
manifest="$inputs/manifests/N1_new_pool_knowledge.json"
admission="$inputs/admissions/N1_new_pool_knowledge.admission.json"
validation="$root/data/development-masked-v2/development-mixture.json"
protocol="$inputs/protocol.json"
parent="$root/posttrain/confirmations-v1/F2_reasoning_no_synthetic-seed20260915-1590004/train/model-final.pt"
parent_sha=f6a4ff43aeeb892ce7dd3dd7ed69f022cef6af688f21b3f005f67850f6041cbc
out="$root/posttrain/f2-fresh-n1-continuation-v1/$arm-seed$seed-$SLURM_JOB_ID"
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python

case "$arm:$peak_lr" in
  f2n1-lr3e5:0.00003|f2n1-lr6e5:0.00006) ;;
  *) echo "arm and learning-rate pair is not frozen" >&2; exit 2 ;;
esac

test -f "$manifest"
test -f "$admission"
test -f "$validation"
test -f "$protocol"
test -f "$parent"
test ! -e "$out"
mkdir -p "$out"
cd "$bundle"
sha256sum -c SHA256SUMS
cd "$src"
sha256sum -c SHA256SUMS
printf '%s  %s\n' "$parent_sha" "$parent" | sha256sum -c -
printf '%s  %s\n' f1cef293be28f9a102ad7d15789f91d242c9c91f5ce7999187d5015ee544f6da "$manifest" | sha256sum -c -
printf '%s  %s\n' f4874ef2adadce7035fa625a3f883fab4397923fd62651a46ce78c136c3a7fd4 "$admission" | sha256sum -c -
printf '%s  %s\n' 495eb7f6d25ccad8fe9f091b9c3033632e464d51f6e59fc056dd6926a47bb819 "$protocol" | sha256sum -c -
printf '%s  %s\n' 429c1ab33bdb8a92214ffdd2327462a05275b29fbd60abecc1d98b9fad634af3 "$validation" | sha256sum -c -

nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv > "$out/gpu.csv"
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA GeForce RTX 5090$')
test "$gpu_count" -eq 4

export P529M_MODEL_SOURCE="$root/source/train-v2"
export PYTHONPATH="$src:$P529M_MODEL_SOURCE"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-f2n1-$SLURM_JOB_ID-inductor"
export TRITON_CACHE_DIR="/tmp/p529m-f2n1-$SLURM_JOB_ID-triton"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1

port=$((29000 + SLURM_JOB_ID % 10000))
srun --nodes=1 --ntasks=1 --cpus-per-task=24 --gres=gpu:4 \
  "$py" -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port="$port" "$src/train_cpt.py" \
  --mixture-manifest "$manifest" --expected-mixture-sha256 f1cef293be28f9a102ad7d15789f91d242c9c91f5ce7999187d5015ee544f6da \
  --validation-manifest "$validation" --expected-validation-sha256 429c1ab33bdb8a92214ffdd2327462a05275b29fbd60abecc1d98b9fad634af3 \
  --validation-tokens 16023552 \
  --admission-receipt "$admission" --expected-admission-sha256 f4874ef2adadce7035fa625a3f883fab4397923fd62651a46ce78c136c3a7fd4 \
  --protocol-file "$protocol" \
  --output-dir "$out/train" \
  --target-tokens 536870912 --pilot-steps 256 \
  --architecture deep --microbatch 4 --accumulation 64 \
  --peak-lr "$peak_lr" --warmup-tokens 67108864 \
  --save-every 999 --validate-every 256 --seed "$seed" \
  --init-model-checkpoint "$parent" --expected-init-sha256 "$parent_sha" \
  --dense-bf16-tflops-per-gpu 209.5 --min-mfu 0.70 \
  --mfu-grace-steps 5 --mfu-window 10 \
  --deterministic --validate-before-training --checkpoint-mode full
