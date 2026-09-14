#!/usr/bin/env bash
#SBATCH --job-name=p529m-eval-v3
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/eval-v3-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/eval-v3-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
export PYTHONPATH="$root/runtime/eval"
export HF_HOME="$root/artifacts/eval-hf-home"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
out="$root/formal/eval-v3"
mkdir -p "$out"
"$py" "$root/source/train-v2/verify_eval_cache.py" \
  --root "$HF_HOME" --manifest "$HF_HOME/eval-cache-manifest.json" \
  | tee "$out/eval-cache-verification.json"
"$py" -m accelerate.commands.launch --multi_gpu --num_processes 4 \
  --main_process_port "$((26000 + SLURM_JOB_ID % 14000))" -m lm_eval \
  --model hf \
  --model_args "pretrained=$root/formal/hf-v3,dtype=bfloat16,trust_remote_code=False,max_length=2048" \
  --tasks hellaswag,piqa,winogrande,openbookqa,arc_easy,arc_challenge,boolq \
  --num_fewshot 0 --batch_size auto --device cuda \
  --output_path "$out" --log_samples --show_config --seed 42,42,42,1234 \
  --metadata '{"protocol_id":"p500m-english-base-v3","protocol_sha256":"38a8adbbba3ca19d2cd59718412fb6671925b6b123350d25a8755eed8661b893"}'
mapfile -t result_files < <(find "$out" -type f -name 'results_*.json' -print)
if [[ "${#result_files[@]}" -ne 1 ]]; then
  printf 'Expected exactly one lm-eval results file, found %s\n' "${#result_files[@]}" >&2
  exit 66
fi
"$py" "$root/source/train-v2/aggregate_eval.py" \
  --protocol "$root/source/protocol-v3.json" \
  --results-json "${result_files[0]}" \
  --output "$out/seven-task-aggregate.json"
printf 'FORMAL_EVAL_V3_COMPLETED\n'
