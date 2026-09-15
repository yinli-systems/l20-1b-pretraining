#!/usr/bin/env bash
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.err
set -euo pipefail

root=/ssd/scxi253/pretrain500m-20260912-v1
bundle="$root/source/mixture-extension-v1"
src="$root/source/cpt-confirmation-v1"
inputs="$root/source/mixture-extension-inputs-v1"
seed=${RUN_SEED:?RUN_SEED must name a frozen extension seed}
case "$seed" in 20260916|20260917) ;; *) echo "unexpected seed: $seed" >&2; exit 64 ;; esac
manifest="$inputs/R4_current_code_guard.json"
admission="$inputs/R4_current_code_guard.admission.json"
protocol="$inputs/protocol.json"
validation="$root/data/development-masked-v2/development-mixture.json"
initial="$root/formal/run-v5/resume.pt"
out="$root/posttrain/mixture-extension-v1/r4-lr6e5-t1b-seed$seed-$SLURM_JOB_ID"
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python

test ! -e "$out"
mkdir -p "$out"
cd "$bundle"
sha256sum -c SHA256SUMS
cd "$src"
sha256sum -c SHA256SUMS
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv > "$out/gpu.csv"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA GeForce RTX 5090$')" -eq 4

mixture_sha=$(sha256sum "$manifest"); mixture_sha=${mixture_sha%% *}
admission_sha=$(sha256sum "$admission"); admission_sha=${admission_sha%% *}
protocol_sha=$(sha256sum "$protocol"); protocol_sha=${protocol_sha%% *}
validation_sha=$(sha256sum "$validation"); validation_sha=${validation_sha%% *}
receipt_sha=$(sha256sum "$inputs/build-receipt.json"); receipt_sha=${receipt_sha%% *}
test "$mixture_sha" = "${EXPECTED_MANIFEST_SHA256:?EXPECTED_MANIFEST_SHA256 is required}"
test "$admission_sha" = "${EXPECTED_ADMISSION_SHA256:?EXPECTED_ADMISSION_SHA256 is required}"
test "$protocol_sha" = "${EXPECTED_PROTOCOL_SHA256:?EXPECTED_PROTOCOL_SHA256 is required}"
test "$receipt_sha" = "${EXPECTED_BUILD_RECEIPT_SHA256:?EXPECTED_BUILD_RECEIPT_SHA256 is required}"

export P529M_MODEL_SOURCE="$root/source/train-v2"
export PYTHONPATH="$src:$P529M_MODEL_SOURCE"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-r4-t1b-$SLURM_JOB_ID-inductor"
export TRITON_CACHE_DIR="/tmp/p529m-r4-t1b-$SLURM_JOB_ID-triton"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1

port=$((26000 + SLURM_JOB_ID % 10000))
srun --nodes=1 --ntasks=1 --cpus-per-task=24 --gres=gpu:4 \
  "$py" -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port="$port" "$src/train_cpt.py" \
  --mixture-manifest "$manifest" --expected-mixture-sha256 "$mixture_sha" \
  --validation-manifest "$validation" --expected-validation-sha256 "$validation_sha" \
  --validation-tokens 16023552 \
  --admission-receipt "$admission" --expected-admission-sha256 "$admission_sha" \
  --protocol-file "$protocol" \
  --output-dir "$out/train" \
  --target-tokens 1073741824 --pilot-steps 512 \
  --architecture deep --microbatch 4 --accumulation 64 \
  --peak-lr 0.00006 --warmup-tokens 67108864 \
  --save-every 999 --validate-every 512 --seed "$seed" \
  --init-model-checkpoint "$initial" \
  --expected-init-sha256 13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf \
  --dense-bf16-tflops-per-gpu 209.5 --min-mfu 0.70 \
  --mfu-grace-steps 5 --mfu-window 10 \
  --deterministic --validate-before-training --checkpoint-mode model-only-final
