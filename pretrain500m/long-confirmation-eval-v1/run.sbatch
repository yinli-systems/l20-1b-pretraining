#!/usr/bin/env bash
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --job-name=p529m-long-confirm-eval
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.err
set -euo pipefail

root=/ssd/scxi253/pretrain500m-20260912-v1
src="$root/source/long-confirmation-eval-v1"
plan="$src/checkpoint-plan.json"
plan_sha=${EXPECTED_CHECKPOINT_PLAN_SHA256:?EXPECTED_CHECKPOINT_PLAN_SHA256 is required}
protocol="$root/source/confirmation-inputs-expansion-v1/confirmation-protocol.json"
protocol_sha=d0c0fbb59bda58de814ab9f297ff4c8f00e9349b8e3f565ce0bb7e0fea4873f5
manifest="$root/data/confirmation-masked-v1/confirmation-mixture.json"
manifest_sha=6b53a58bbfe8991ecbbd754e8e27c3d832410f353a69f44a6180eb1e78624ae2
out="$root/posttrain/long-confirmation-eval-v1/$SLURM_JOB_ID"
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python

test ! -e "$out"
mkdir -p "$out"
cd "$src"
sha256sum -c SHA256SUMS
echo "$plan_sha  $plan" | sha256sum -c -
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv > "$out/gpu.csv"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA GeForce RTX 5090$')" -eq 4
test "$(tail -n +2 "$out/gpu.csv" | cut -d, -f4 | sort -u | wc -l)" -eq 4

export P529M_MODEL_SOURCE="$root/source/train-v2"
export PYTHONPATH="$src:$P529M_MODEL_SOURCE"
export OMP_NUM_THREADS=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((30000 + SLURM_JOB_ID % 10000))
export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-long-confirm-eval-$SLURM_JOB_ID-inductor"
export TRITON_CACHE_DIR="/tmp/p529m-long-confirm-eval-$SLURM_JOB_ID-triton"

srun --nodes=1 --ntasks=1 --cpus-per-task=24 --gres=gpu:4 \
  "$py" -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" "$src/evaluate.py" \
  --validation-manifest "$manifest" --expected-validation-sha256 "$manifest_sha" \
  --validation-tokens 16678912 --microbatch 4 \
  --protocol-file "$protocol" --expected-protocol-sha256 "$protocol_sha" \
  --checkpoint-plan "$plan" --expected-checkpoint-plan-sha256 "$plan_sha" \
  --output "$out/result.json"

result_sha=$(sha256sum "$out/result.json"); result_sha=${result_sha%% *}
"$py" "$src/select_long.py" \
  --evaluation "$out/result.json" --expected-evaluation-sha256 "$result_sha" \
  --protocol "$protocol" --expected-protocol-sha256 "$protocol_sha" \
  --output "$out/selection.json"
sha256sum "$out/result.json" "$out/selection.json" > "$out/SHA256SUMS"
