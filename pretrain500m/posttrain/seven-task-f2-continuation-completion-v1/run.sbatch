#!/usr/bin/env bash
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --job-name=p529m-f2-seven-complete
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.err
#SBATCH --exclude=wqd10nbl11g5,wqd10nah08g5
set -euo pipefail

root=/ssd/scxi253/pretrain500m-20260912-v1
src="$root/source/seven-task-f2-continuation-completion-v1"
plan="$src/plan.json"
plan_sha=76b0f5bbe5581f332d7e4b386d0acf34b91f3f5dc5e9b70602f5a923568c04fc
train_src="$root/source/train-v2"
protocol="$root/source/protocol-v5.json"
protocol_sha=a71ce3c8cc85cac6fe6751cb46f7e50dfb78d266602d6377cb2e65249f11c761
cache_root="$root/artifacts/eval-hf-home"
cache_manifest="$cache_root/eval-cache-manifest.json"
cache_manifest_sha=699558cb80318870306eccf77fb1d200413bfdc03b6548fe41c9d39ed0044620
tokenizer_dir="$root/source/pack-v3/tokenizer"
parent_checkpoint="$root/posttrain/confirmations-v1/F2_reasoning_no_synthetic-seed20260915-1590004/train/model-final.pt"
parent_checkpoint_sha=f6a4ff43aeeb892ce7dd3dd7ed69f022cef6af688f21b3f005f67850f6041cbc
parent_receipt_sha=14ed94adbef18e4ee898806bd9ad96ab79306283676fb22270aafe2e73c78d33
parent_model_sha=eff69f11a24eeebf248958ecbc870c6a5ae5e54c02e453b1e8fc0082a8aab9b5
continuation_export="$root/posttrain/f2-export-margin-aware-v1/1593643/task-1/F2_continuation_seed20260915-hf"
continuation_receipt_sha=be0bb72517d4df1e0a6df9845e0c7f53139898ab23085b7597b6316937689abd
continuation_model_sha=803022851e95a852d76ce08240fec07fecc0d987158d380d434fb40e54d1eb9b
out="$root/posttrain/seven-task-f2-continuation-completion-v1/$SLURM_JOB_ID"
tmp_root="/tmp/p529m-f2-seven-complete-$SLURM_JOB_ID"
parent_export="$tmp_root/F2_parent_seed20260915"
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

verify_export() {
  local export_dir=$1
  local expected_receipt_sha=$2
  local expected_model_sha=$3
  echo "$expected_receipt_sha  $export_dir/export-receipt.json" | sha256sum -c -
  "$py" - "$export_dir" "$expected_model_sha" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); expected_model=sys.argv[2]
receipt=json.loads((root/'export-receipt.json').read_text())
for item in receipt['files']:
    path=root/item['path']; digest=hashlib.sha256()
    if path.stat().st_size != item['bytes']:
        raise SystemExit(f"size mismatch: {path}")
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16*1024*1024),b''):
            digest.update(chunk)
    value=digest.hexdigest()
    if value != item['sha256']:
        raise SystemExit(f"hash mismatch: {path}")
    if path.name=='model.safetensors' and value != expected_model:
        raise SystemExit(f"unexpected model identity: {value}")
PY
}

evaluate_hf() {
  local candidate_id=$1
  local export_dir=$2
  local checkpoint_sha=$3
  local evaluation_index=$4
  local candidate_out="$out/$candidate_id"
  local lm_out="$candidate_out/lm-eval"
  local metadata
  mkdir -p "$candidate_out"
  metadata=$(printf '{"protocol_id":"p500m-english-base-v5","candidate_id":"%s","checkpoint_sha256":"%s","world_size":2,"evaluation_class":"adaptive_f2_continuation_completion"}' "$candidate_id" "$checkpoint_sha")
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
}

echo "$parent_checkpoint_sha  $parent_checkpoint" | sha256sum -c -
"$py" "$train_src/export_hf.py" \
  --checkpoint "$parent_checkpoint" --tokenizer-dir "$tokenizer_dir" \
  --output-dir "$parent_export" --expected-step 1024
verify_export "$parent_export" "$parent_receipt_sha" "$parent_model_sha"
mkdir -p "$out/F2_parent_seed20260915"
cp "$parent_export/export-receipt.json" "$out/F2_parent_seed20260915/export-receipt.json"
evaluate_hf F2_parent_seed20260915 "$parent_export" "$parent_checkpoint_sha" 0

verify_export "$continuation_export" "$continuation_receipt_sha" "$continuation_model_sha"
mkdir -p "$out/F2_continuation_seed20260915"
cp "$continuation_export/export-receipt.json" "$out/F2_continuation_seed20260915/export-receipt.json"
evaluate_hf F2_continuation_seed20260915 "$continuation_export" 86ab770e671dd845c6de1c7724fb90cc22bc79449f9133e2a01047814f86fed8 1

"$py" "$src/summarize.py" --plan "$plan" \
  --new-parent "$out/F2_parent_seed20260915/seven-task-aggregate.json" \
  --new-continuation "$out/F2_continuation_seed20260915/seven-task-aggregate.json" \
  --output "$out/summary.json"
sha256sum "$out/summary.json" > "$out/SHA256SUMS"
cleanup
test ! -e "$tmp_root"
trap - EXIT
printf 'F2_SEVEN_TASK_COMPLETION_DONE summary_sha256=%s\n' "$(sha256sum "$out/summary.json" | cut -d' ' -f1)"
