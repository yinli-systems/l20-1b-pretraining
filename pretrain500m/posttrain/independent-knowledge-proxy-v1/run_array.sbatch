#!/usr/bin/env bash
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --array=0-2%3
#SBATCH --time=01:00:00
#SBATCH --job-name=p529m-sciq-proxy-1g
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%A_%a.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%A_%a.err
set -euo pipefail

root=/ssd/scxi253/pretrain500m-20260912-v1
src="$root/source/independent-knowledge-proxy-v1"
plan="$src/plan.json"
plan_sha=756beb07408801f4b8ad6888306c9f0f5afb8a2f6d1eb44371e31120cc8acb2e
splits="$src/splits.json"
data="$src/sciq-test.jsonl"
tokenizer_dir="$root/source/pack-v3/tokenizer"
train_src="$root/source/train-v2"
out="$root/posttrain/independent-knowledge-proxy-v1/$SLURM_ARRAY_JOB_ID"
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
candidate_ids=(base F2_parent_seed20260914 F2_parent_seed20260915)
candidate_id="${candidate_ids[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "$out"
test ! -e "$out/$candidate_id.json"
cd "$src"
sha256sum -c SHA256SUMS
echo "$plan_sha  $plan" | sha256sum -c -
unshare -Urn true
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv > "$out/$candidate_id.gpu.csv"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA GeForce RTX 4090$')" -eq 1
test "$(df -B1 --output=avail "$out" | tail -n 1)" -gt 4294967296

export PYTHONPATH="$root/runtime/eval:$train_src"
export HF_HOME="$root/artifacts/eval-hf-home"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$py" "$src/evaluate.py" \
  --plan "$plan" --candidate-id "$candidate_id" --data "$data" \
  --splits "$splits" --tokenizer-dir "$tokenizer_dir" \
  --role development --output "$out/$candidate_id.json" \
  > "$out/$candidate_id.stdout" 2> "$out/$candidate_id.stderr"

exec 9>"$out/.summarize.lock"
flock 9
if test -f "$out/base.json" \
  && test -f "$out/F2_parent_seed20260914.json" \
  && test -f "$out/F2_parent_seed20260915.json" \
  && test ! -f "$out/summary.json"; then
  "$py" "$src/summarize.py" --plan "$plan" \
    --result "$out/base.json" --result "$out/F2_parent_seed20260914.json" \
    --result "$out/F2_parent_seed20260915.json" --output "$out/summary.json"
  sha256sum "$out"/*.json "$out"/*.stdout "$out"/*.stderr \
    "$out"/*.gpu.csv > "$out/SHA256SUMS"
  printf 'INDEPENDENT_KNOWLEDGE_PROXY_V1_DEVELOPMENT_COMPLETE summary_sha256=%s\n' \
    "$(sha256sum "$out/summary.json" | cut -d' ' -f1)"
fi
