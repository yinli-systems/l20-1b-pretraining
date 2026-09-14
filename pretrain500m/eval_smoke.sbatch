#!/usr/bin/env bash
#SBATCH --job-name=p529m-eval-smoke
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:20:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/eval-smoke-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/eval-smoke-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
export PYTHONPATH="$root/runtime/eval"
export HF_HOME="$root/artifacts/eval-hf-home"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
out="$root/pilots/eval-smoke-$SLURM_JOB_ID"
mkdir -p "$out"
"$py" "$root/source/train-v2/verify_eval_cache.py" \
  --root "$HF_HOME" --manifest "$HF_HOME/eval-cache-manifest.json" \
  | tee "$out/eval-cache-verification.json"
"$py" -m lm_eval --model hf \
  --model_args "pretrained=$root/pilots/eval-smoke-model,dtype=bfloat16,trust_remote_code=False,max_length=2048" \
  --tasks hellaswag,piqa,winogrande,openbookqa,arc_easy,arc_challenge,boolq \
  --num_fewshot 0 --batch_size 2 --device cuda --limit 2 \
  --output_path "$out" --log_samples --show_config --seed 42,42,42,1234 \
  --metadata '{"protocol_id":"p500m-english-base-v2","smoke_only":true}'
printf 'EVAL_SMOKE_COMPLETED\n'
