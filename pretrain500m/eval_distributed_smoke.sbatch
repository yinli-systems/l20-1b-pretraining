#!/usr/bin/env bash
#SBATCH --job-name=p529m-eval-dist-smoke-v2
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/eval-dist-smoke-v2-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/eval-dist-smoke-v2-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
model=${P500M_SMOKE_MODEL:?P500M_SMOKE_MODEL is required}
export PYTHONPATH="$root/runtime/eval"
export HF_HOME="$root/artifacts/eval-hf-home"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
out="$root/pilots/eval-dist-smoke-v2-$SLURM_JOB_ID"
mkdir -p "$out"
"$py" "$root/source/train-v2/verify_eval_cache.py" \
  --root "$HF_HOME" --manifest "$HF_HOME/eval-cache-manifest.json" \
  | tee "$out/eval-cache-verification.json"
"$py" -m accelerate.commands.launch --multi_gpu --num_processes 4 \
  --main_process_port "$((26000 + SLURM_JOB_ID % 14000))" -m lm_eval \
  --model hf --model_args "pretrained=$model,dtype=bfloat16,trust_remote_code=False,max_length=2048" \
  --tasks hellaswag,piqa,winogrande,openbookqa,arc_easy,arc_challenge,boolq \
  --num_fewshot 0 --batch_size auto --device cuda --limit 8 \
  --output_path "$out" --log_samples --show_config --seed 42,42,42,1234 \
  --metadata '{"protocol_id":"p500m-english-base-v2","distributed_smoke_only":true}'
printf 'DISTRIBUTED_EVAL_SMOKE_V2_COMPLETED model=%s\n' "$model"
