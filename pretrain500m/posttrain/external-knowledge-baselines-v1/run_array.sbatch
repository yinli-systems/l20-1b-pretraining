#!/usr/bin/env bash
#SBATCH --partition=gpu_4090,gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --array=0-3%4
#SBATCH --time=01:00:00
#SBATCH --job-name=p529m-ext-sciq-1g
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%A_%a.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%A_%a.err
#SBATCH --exclude=wqd10nbl11g5,wqd10nah08g5
set -euo pipefail

root=/ssd/scxi253/pretrain500m-20260912-v1
src="$root/source/external-knowledge-baselines-v1"
plan="$src/plan.json"
data="$root/source/independent-knowledge-proxy-v1/sciq-test.jsonl"
splits="$root/source/independent-knowledge-proxy-v1/splits.json"
models="$root/artifacts/external-knowledge-baselines-v1"
base="$root/posttrain/independent-knowledge-proxy-v1/1594381/base.json"
out="$root/posttrain/external-knowledge-baselines-v1/$SLURM_ARRAY_JOB_ID"
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
candidate_ids=(qwen2.5-0.5b openelm-450m smollm2-360m mobilellm-600m)
candidate_id="${candidate_ids[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "$out"
test ! -e "$out/$candidate_id.json"
cd "$src"
sha256sum -c SHA256SUMS
unshare -Urn true
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv > "$out/$candidate_id.gpu.csv"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
test "$gpu_name" = "NVIDIA GeForce RTX 4090" || test "$gpu_name" = "NVIDIA GeForce RTX 5090"
test "$(df -B1 --output=avail "$out" | tail -n 1)" -gt 21474836480

export HF_HOME="$root/artifacts/eval-hf-home"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$py" "$src/evaluate.py" \
  --plan "$plan" --candidate-id "$candidate_id" --data "$data" --splits "$splits" \
  --model-root "$models" --output "$out/$candidate_id.json" \
  > "$out/$candidate_id.stdout" 2> "$out/$candidate_id.stderr"

exec 9>"$out/.summarize.lock"
flock 9
if test -f "$out/qwen2.5-0.5b.json" \
  && test -f "$out/openelm-450m.json" \
  && test -f "$out/smollm2-360m.json" \
  && test -f "$out/mobilellm-600m.json" \
  && test ! -f "$out/summary.json"; then
  "$py" "$src/summarize.py" --plan "$plan" --base "$base" \
    --result "$out/qwen2.5-0.5b.json" --result "$out/openelm-450m.json" \
    --result "$out/smollm2-360m.json" --result "$out/mobilellm-600m.json" \
    --output "$out/summary.json"
  sha256sum "$out"/*.json "$out"/*.stdout "$out"/*.stderr "$out"/*.gpu.csv > "$out/SHA256SUMS"
  printf 'MATCHED_EXTERNAL_CLOSED_BOOK_SCIQ_COMPLETE summary_sha256=%s\n' \
    "$(sha256sum "$out/summary.json" | cut -d' ' -f1)"
fi
