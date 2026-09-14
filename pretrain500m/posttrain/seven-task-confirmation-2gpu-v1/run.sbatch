#!/usr/bin/env bash
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=02:30:00
#SBATCH --job-name=p529m-seven-task-2gpu
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.err
#SBATCH --exclude=wqd10nbl11g5,wqd10nah08g5
set -euo pipefail

root=/ssd/scxi253/pretrain500m-20260912-v1
src="$root/source/seven-task-confirmation-2gpu-v1"
plan="$src/candidates.json"
plan_sha=${EXPECTED_PLAN_SHA256:?EXPECTED_PLAN_SHA256 is required}
train_src="$root/source/train-v2"
protocol="$root/source/protocol-v5.json"
protocol_sha=a71ce3c8cc85cac6fe6751cb46f7e50dfb78d266602d6377cb2e65249f11c761
cache_root="$root/artifacts/eval-hf-home"
cache_manifest="$cache_root/eval-cache-manifest.json"
cache_manifest_sha=699558cb80318870306eccf77fb1d200413bfdc03b6548fe41c9d39ed0044620
reference_base="$root/formal/eval-v5/seven-task-aggregate.json"
reference_base_sha=34265389f4f85d54a1cdb4945b06016849f58b55b6c47e686ecc7c2423fd8c09
base_checkpoint="$root/formal/run-v5/resume.pt"
base_checkpoint_sha=13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf
base_export="$root/formal/hf-v5"
base_export_receipt_sha=736705f36daad6a405e7a604191673ce43077ede464841f37888e985c891d0e4
tokenizer_dir="$root/source/pack-v3/tokenizer"
out="$root/posttrain/seven-task-confirmation-2gpu-v1/$SLURM_JOB_ID"
tmp_root="/tmp/p529m-seven-task-2gpu-$SLURM_JOB_ID"
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python

test ! -e "$out"
test ! -e "$tmp_root"
mkdir -p "$out" "$tmp_root"
cleanup() { rm -rf -- "$tmp_root"; }
trap cleanup EXIT
cd "$src"
sha256sum -c SHA256SUMS
echo "$plan_sha  $plan" | sha256sum -c -
echo "$protocol_sha  $protocol" | sha256sum -c -
echo "$cache_manifest_sha  $cache_manifest" | sha256sum -c -
echo "$reference_base_sha  $reference_base" | sha256sum -c -

nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv > "$out/gpu.csv"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA GeForce RTX 5090$')" -eq 2
test "$(tail -n +2 "$out/gpu.csv" | cut -d, -f4 | sort -u | wc -l)" -eq 2
test "$(df -B1 --output=avail "$tmp_root" | tail -n 1)" -gt 4294967296
test "$(df -B1 --output=avail "$out" | tail -n 1)" -gt 2147483648

export PYTHONPATH="$root/runtime/eval:$train_src"
export HF_HOME="$cache_root"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$py" "$train_src/verify_eval_cache.py" --root "$HF_HOME" --manifest "$cache_manifest" \
  | tee "$out/eval-cache-verification.json"

evaluate_hf() {
  local candidate_id=$1
  local export_dir=$2
  local checkpoint_sha=$3
  local candidate_out="$out/$candidate_id"
  local lm_out="$candidate_out/lm-eval"
  local metadata
  mkdir -p "$candidate_out"
  metadata=$(printf '{"protocol_id":"p500m-english-base-v5","candidate_id":"%s","checkpoint_sha256":"%s","world_size":2}' "$candidate_id" "$checkpoint_sha")
  "$py" -m accelerate.commands.launch --multi_gpu --num_processes 2 \
    --main_process_port "$((26000 + SLURM_JOB_ID % 13000 + evaluation_index))" -m lm_eval \
    --model hf \
    --model_args "pretrained=$export_dir,dtype=bfloat16,trust_remote_code=False,max_length=2048" \
    --tasks hellaswag,piqa,winogrande,openbookqa,arc_easy,arc_challenge,boolq \
    --num_fewshot 0 --batch_size auto --device cuda \
    --output_path "$lm_out" --log_samples --show_config --seed 42,42,42,1234 \
    --metadata "$metadata"
  mapfile -t result_files < <(find "$lm_out" -type f -name 'results_*.json' -print)
  test "${#result_files[@]}" -eq 1
  "$py" "$train_src/aggregate_eval.py" \
    --protocol "$protocol" --results-json "${result_files[0]}" \
    --output "$candidate_out/seven-task-aggregate.json"
  evaluation_index=$((evaluation_index + 1))
}

echo "$base_checkpoint_sha  $base_checkpoint" | sha256sum -c -
echo "$base_export_receipt_sha  $base_export/export-receipt.json" | sha256sum -c -
"$py" - "$base_export" <<'PY'
import hashlib, json, pathlib, sys
root=pathlib.Path(sys.argv[1])
receipt=json.loads((root/'export-receipt.json').read_text())
for item in receipt['files']:
    path=root/item['path']
    if path.stat().st_size != item['bytes']:
        raise SystemExit(f"size mismatch: {path}")
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    value=digest.hexdigest()
    if value != item['sha256']:
        raise SystemExit(f"hash mismatch: {path}")
PY
mkdir -p "$out/base"
cp "$base_export/export-receipt.json" "$out/base/export-receipt.json"
evaluation_index=0
evaluate_hf base "$base_export" "$base_checkpoint_sha"

mapfile -t candidates < <("$py" - "$plan" <<'PY'
import json, sys
document=json.load(open(sys.argv[1]))
for item in document['candidates']:
    print('\t'.join(str(item[key]) for key in ('id','checkpoint','checkpoint_sha256','expected_step')))
PY
)
test "${#candidates[@]}" -eq 4

for record in "${candidates[@]}"; do
  IFS=$'\t' read -r candidate_id checkpoint checkpoint_sha expected_step <<< "$record"
  candidate_out="$out/$candidate_id"
  export_dir="$tmp_root/$candidate_id"
  mkdir -p "$candidate_out"
  echo "$checkpoint_sha  $checkpoint" | sha256sum -c -
  "$py" "$train_src/export_hf.py" \
    --checkpoint "$checkpoint" --tokenizer-dir "$tokenizer_dir" \
    --output-dir "$export_dir" --expected-step "$expected_step"
  cp "$export_dir/export-receipt.json" "$candidate_out/export-receipt.json"
  evaluate_hf "$candidate_id" "$export_dir" "$checkpoint_sha"
  rm -rf -- "$export_dir"
  test ! -e "$export_dir"
done

"$py" "$src/summarize.py" --plan "$plan" \
  --base "$out/base/seven-task-aggregate.json" --reference-base "$reference_base" \
  --results-root "$out" --output "$out/summary.json"
sha256sum "$out/summary.json" > "$out/SHA256SUMS"
cleanup
test ! -e "$tmp_root"
trap - EXIT
printf 'SEVEN_TASK_2GPU_CONFIRMATION_COMPLETED summary_sha256=%s\n' "$(sha256sum "$out/summary.json" | cut -d' ' -f1)"
