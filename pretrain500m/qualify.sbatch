#!/usr/bin/env bash
#SBATCH --job-name=p500m-qualify-v1
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/qualify-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/qualify-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/overlay:$root/source/qualification-v1"
export HF_HOME="$root/data/hf-cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="/tmp/p500m-$SLURM_JOB_ID-inductor"
export TRITON_CACHE_DIR="/tmp/p500m-$SLURM_JOB_ID-triton"
out="$root/receipts/qualification-$SLURM_JOB_ID"
mkdir "$out"
cd "$root/source/qualification-v1"
date -u
hostname
printf 'JOB=%s CUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,power.limit,clocks.max.sm --format=csv > "$out/gpu.csv"
nvidia-smi topo -m > "$out/topology.txt"
df -h /tmp /ssd/scxi253 > "$out/storage.txt"
"$py" -m pip freeze > "$out/runtime-freeze.txt"
"$py" -m pytest -q -p no:cacheprovider test_model.py > "$out/model-tests.txt" 2>&1
"$py" corpus_audit.py --root /ssd/scxi253/corpus/fineweb-edu-10BT --manifest sources/fineweb-edu-10bt.json --output "$out/corpus-audit.json" > "$out/corpus-audit.log" 2>&1 &
audit_pid=$!
"$py" gpu_probe.py --microbatch 2 --output "$out/single-eager-mb2.json" > "$out/single-eager-mb2.log" 2>&1
"$py" -m torch.distributed.run --standalone --nproc_per_node=4 gpu_probe.py --microbatch 2 --output "$out/four-eager-mb2.json" > "$out/four-eager-mb2.log" 2>&1
"$py" gpu_probe.py --compile --microbatch 2 --output "$out/single-compile-mb2.json" > "$out/single-compile-mb2.log" 2>&1
"$py" -m torch.distributed.run --standalone --nproc_per_node=4 gpu_probe.py --compile --microbatch 2 --output "$out/four-compile-mb2.json" > "$out/four-compile-mb2.log" 2>&1
wait "$audit_pid"
printf 'QUALIFICATION_STAGES_COMPLETED\n'
