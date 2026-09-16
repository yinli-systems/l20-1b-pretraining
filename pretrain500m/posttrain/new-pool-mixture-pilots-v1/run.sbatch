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
bundle="$root/source/new-pool-mixture-pilots-v1"
src="$root/source/cpt-confirmation-v1"
inputs="$root/source/new-pool-mixture-pilot-inputs-v1"
recipe=${RECIPE:?RECIPE must name the frozen pilot recipe}
arm=${ARM_ID:?ARM_ID must name the frozen pilot arm}
seed=${RUN_SEED:?RUN_SEED must name the frozen pilot seed}
peak_lr=${PEAK_LR:?PEAK_LR must name the frozen pilot learning rate}
manifest="$inputs/manifests/$recipe.json"
admission="$inputs/admissions/$recipe.admission.json"
validation="$root/data/development-masked-v2/development-mixture.json"
protocol="$inputs/protocol.json"
initial="$root/formal/run-v5/resume.pt"
out="$root/posttrain/new-pool-mixture-pilots-v1/$arm-seed$seed-$SLURM_JOB_ID"
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python

test -f "$manifest"
test -f "$admission"
test ! -e "$out"
mkdir -p "$out"
cd "$bundle"
sha256sum -c SHA256SUMS
cd "$src"
sha256sum -c SHA256SUMS
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv > "$out/gpu.csv"
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA GeForce RTX 5090$')
test "$gpu_count" -eq 4

mixture_sha=$(sha256sum "$manifest"); mixture_sha=${mixture_sha%% *}
admission_sha=$(sha256sum "$admission"); admission_sha=${admission_sha%% *}
validation_sha=$(sha256sum "$validation"); validation_sha=${validation_sha%% *}

export P529M_MODEL_SOURCE="$root/source/train-v2"
export PYTHONPATH="$src:$P529M_MODEL_SOURCE"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-new-pool-$SLURM_JOB_ID-inductor"
export TRITON_CACHE_DIR="/tmp/p529m-new-pool-$SLURM_JOB_ID-triton"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1

port=$((28000 + SLURM_JOB_ID % 10000))
srun --nodes=1 --ntasks=1 --cpus-per-task=24 --gres=gpu:4 \
  "$py" -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port="$port" "$src/train_cpt.py" \
  --mixture-manifest "$manifest" --expected-mixture-sha256 "$mixture_sha" \
  --validation-manifest "$validation" --expected-validation-sha256 "$validation_sha" \
  --validation-tokens 16023552 \
  --admission-receipt "$admission" --expected-admission-sha256 "$admission_sha" \
  --protocol-file "$protocol" \
  --output-dir "$out/train" \
  --target-tokens 536870912 --pilot-steps 256 \
  --architecture deep --microbatch 4 --accumulation 64 \
  --peak-lr "$peak_lr" --warmup-tokens 67108864 \
  --save-every 999 --validate-every 256 --seed "$seed" \
  --init-model-checkpoint "$initial" \
  --expected-init-sha256 13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf \
  --dense-bf16-tflops-per-gpu 209.5 --min-mfu 0.70 \
  --mfu-grace-steps 5 --mfu-window 10 \
  --deterministic --validate-before-training --checkpoint-mode model-only-final
