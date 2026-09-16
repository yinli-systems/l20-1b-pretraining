#!/usr/bin/env bash
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --job-name=p529m-code-proxy
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.err
#SBATCH --exclude=wqd10nbl11g5,wqd10nah08g5
set -euo pipefail

root=/ssd/scxi253/pretrain500m-20260912-v1
src="$root/source/independent-code-proxy-v1"
plan="$src/plan.json"
plan_sha=a7e7cc4bf3f76bf1b255229fe9e5db6e6352bc94ad250a17af6baf72f488bb6f
splits="$src/splits.json"
rows="$src/rows.jsonl"
data="$root/evaluation/benchmark-reservation-v1/data/raw/huggingface/google-research-datasets/mbpp/sanitized/test-00000-of-00001.parquet"
tokenizer_dir="$root/source/pack-v3/tokenizer"
train_src="$root/source/train-v2"
out="$root/posttrain/independent-code-proxy-v1/$SLURM_JOB_ID"
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python

test ! -e "$out"
mkdir -p "$out"
cd "$src"
sha256sum -c SHA256SUMS
echo "$plan_sha  $plan" | sha256sum -c -
echo 'e9e9efa2c0d59ef5e55537a9d126b8f875d5ac010a8d75628d76824884e15850'"  $data" | sha256sum -c -
unshare -Urn true
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv > "$out/gpu.csv"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA GeForce RTX 5090$')" -eq 3
test "$(tail -n +2 "$out/gpu.csv" | cut -d, -f4 | sort -u | wc -l)" -eq 3
test "$(df -B1 --output=avail "$out" | tail -n 1)" -gt 4294967296

export PYTHONPATH="$root/runtime/eval:$train_src"
export HF_HOME="$root/artifacts/eval-hf-home"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

candidate_ids=(base F2_parent_seed20260914 F2_parent_seed20260915)
pids=()
for index in 0 1 2; do
  candidate_id="${candidate_ids[$index]}"
  CUDA_VISIBLE_DEVICES="$index" "$py" "$src/evaluate.py" \
    --plan "$plan" --candidate-id "$candidate_id" --rows "$rows" \
    --splits "$splits" --tokenizer-dir "$tokenizer_dir" \
    --role development --output "$out/$candidate_id.json" \
    > "$out/$candidate_id.stdout" 2> "$out/$candidate_id.stderr" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
test "$status" -eq 0

"$py" "$src/summarize.py" --plan "$plan" \
  --result "$out/base.json" --result "$out/F2_parent_seed20260914.json" \
  --result "$out/F2_parent_seed20260915.json" --output "$out/summary.json"
sha256sum "$out"/*.json "$out"/*.stdout "$out"/*.stderr "$out/gpu.csv" > "$out/SHA256SUMS"
printf 'INDEPENDENT_CODE_PROXY_V1_DEVELOPMENT_COMPLETE summary_sha256=%s\n' \
  "$(sha256sum "$out/summary.json" | cut -d' ' -f1)"
