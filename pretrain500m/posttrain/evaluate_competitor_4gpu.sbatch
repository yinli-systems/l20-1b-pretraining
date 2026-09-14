#!/usr/bin/env bash
#SBATCH --job-name=p529m-competitor-eval
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/competitor-eval-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/competitor-eval-%j.err
set -euo pipefail
: "${P529M_MODEL_SLUG:?P529M_MODEL_SLUG is required}"
: "${P529M_MODEL_REVISION:?P529M_MODEL_REVISION is required}"
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
model="$root/baselines/models/$P529M_MODEL_SLUG"
out="$root/baselines/eval/$P529M_MODEL_SLUG"
if [[ -e "$out" ]] && find "$out" -mindepth 1 -print -quit | grep -q .; then
  printf 'Output directory is not empty: %s\n' "$out" >&2
  exit 73
fi
mkdir -p "$out"
export PYTHONPATH="$root/runtime/eval"
export HF_HOME="$root/artifacts/eval-hf-home"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
protocol_sha=$(sha256sum "$root/source/protocol-v5.json" | awk '{print $1}')
metadata=$(printf '{"protocol_id":"p500m-english-base-v5","protocol_sha256":"%s","competitor_slug":"%s","competitor_revision":"%s"}' "$protocol_sha" "$P529M_MODEL_SLUG" "$P529M_MODEL_REVISION")
"$py" "$root/source/train-v2/verify_eval_cache.py" \
  --root "$HF_HOME" --manifest "$HF_HOME/eval-cache-manifest.json" \
  | tee "$out/eval-cache-verification.json"
"$py" -m accelerate.commands.launch --multi_gpu --num_processes 4 \
  --main_process_port "$((26000 + SLURM_JOB_ID % 14000))" -m lm_eval \
  --model hf \
  --model_args "pretrained=$model,dtype=bfloat16,trust_remote_code=False,max_length=2048" \
  --tasks hellaswag,piqa,winogrande,openbookqa,arc_easy,arc_challenge,boolq \
  --num_fewshot 0 --batch_size auto --device cuda \
  --output_path "$out" --log_samples --show_config --seed 42,42,42,1234 \
  --metadata "$metadata"
mapfile -t result_files < <(find "$out" -type f -name 'results_*.json' -print)
if [[ "${#result_files[@]}" -ne 1 ]]; then
  printf 'Expected exactly one lm-eval results file, found %s\n' "${#result_files[@]}" >&2
  exit 66
fi
"$py" "$root/source/train-v2/aggregate_eval.py" \
  --protocol "$root/source/protocol-v5.json" \
  --results-json "${result_files[0]}" \
  --output "$out/seven-task-aggregate.json"
printf 'COMPETITOR_EVAL_COMPLETED slug=%s revision=%s protocol_sha=%s\n' \
  "$P529M_MODEL_SLUG" "$P529M_MODEL_REVISION" "$protocol_sha"
