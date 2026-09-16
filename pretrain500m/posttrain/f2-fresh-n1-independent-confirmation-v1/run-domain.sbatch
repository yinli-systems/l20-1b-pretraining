#!/usr/bin/env bash
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.err
set -euo pipefail

root=/ssd/scxi253/pretrain500m-20260912-v1
src="$root/source/f2-fresh-n1-independent-confirmation-v1"
domain=${DOMAIN:?DOMAIN is required}
expected_bundle_sha=${EXPECTED_BUNDLE_SHA256:?EXPECTED_BUNDLE_SHA256 is required}
case "$domain" in
  reading|math|code|knowledge) ;;
  *) echo "unsupported domain: $domain" >&2; exit 2 ;;
esac

domain_src="$src/$domain"
plan="$domain_src/plan.json"
splits="$domain_src/splits.json"
tokenizer_dir="$root/source/pack-v3/tokenizer"
train_src="$root/source/train-v2"
out="$root/posttrain/f2-fresh-n1-independent-confirmation-v1/$domain-$SLURM_JOB_ID"
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python

test ! -e "$out"
mkdir -p "$out"
cd "$src"
sha256sum -c SHA256SUMS
echo "$expected_bundle_sha  $src/SHA256SUMS" | sha256sum -c -

nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv > "$out/gpu.csv"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA GeForce RTX 4090$')" -eq 2
test "$(tail -n +2 "$out/gpu.csv" | cut -d, -f4 | sort -u | wc -l)" -eq 2
test "$(df -B1 --output=avail "$out" | tail -n 1)" -gt 4294967296

export PYTHONPATH="$root/runtime/eval:$train_src"
export HF_HOME="$root/artifacts/eval-hf-home"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

candidate_ids=(F2_parent_seed20260915 f2n1-lr3e5)
pids=()
for index in 0 1; do
  candidate_id="${candidate_ids[$index]}"
  common=(--plan "$plan" --candidate-id "$candidate_id" --splits "$splits" \
    --tokenizer-dir "$tokenizer_dir" --role confirmation \
    --output "$out/$candidate_id.json")
  case "$domain" in
    reading)
      data="$root/evaluation/benchmark-reservation-v1/data/raw/huggingface/facebook/belebele/data/eng_Latn.jsonl"
      CUDA_VISIBLE_DEVICES="$index" "$py" "$domain_src/evaluate.py" "${common[@]}" \
        --data "$data" > "$out/$candidate_id.stdout" 2> "$out/$candidate_id.stderr" &
      ;;
    math)
      data="$root/evaluation/benchmark-reservation-v1/data/raw/github/google-research/url-nlp/mgsm/mgsm_en.tsv"
      CUDA_VISIBLE_DEVICES="$index" "$py" "$domain_src/evaluate.py" "${common[@]}" \
        --data "$data" > "$out/$candidate_id.stdout" 2> "$out/$candidate_id.stderr" &
      ;;
    code)
      unshare -Urn true
      CUDA_VISIBLE_DEVICES="$index" "$py" "$domain_src/evaluate.py" "${common[@]}" \
        --rows "$domain_src/rows.jsonl" > "$out/$candidate_id.stdout" 2> "$out/$candidate_id.stderr" &
      ;;
    knowledge)
      data="$root/source/independent-knowledge-proxy-v1/sciq-test.jsonl"
      CUDA_VISIBLE_DEVICES="$index" "$py" "$domain_src/evaluate.py" "${common[@]}" \
        --data "$data" > "$out/$candidate_id.stdout" 2> "$out/$candidate_id.stderr" &
      ;;
  esac
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
test "$status" -eq 0

"$py" "$src/summarize_domain.py" --domain "$domain" --plan "$plan" \
  --parent "$out/F2_parent_seed20260915.json" \
  --candidate "$out/f2n1-lr3e5.json" --output "$out/summary.json"
sha256sum "$out"/*.json "$out"/*.stdout "$out"/*.stderr "$out/gpu.csv" > "$out/SHA256SUMS"
printf 'F2_FRESH_N1_INDEPENDENT_CONFIRMATION_COMPLETE domain=%s summary_sha256=%s\n' \
  "$domain" "$(sha256sum "$out/summary.json" | cut -d' ' -f1)"
