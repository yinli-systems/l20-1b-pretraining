#!/usr/bin/env bash
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --job-name=p529m-confirm-eval8
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.err
set -euo pipefail

root=/ssd/scxi253/pretrain500m-20260912-v1
src="$root/source/confirmation-eval-v1"
out="$root/posttrain/confirmation-eval-v1/$SLURM_JOB_ID"
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
manifest="$root/data/confirmation-masked-v1/confirmation-mixture.json"
mkdir -p "$out"
cd "$src"
sha256sum -c SHA256SUMS
export P529M_MODEL_SOURCE="$root/source/train-v2"
export PYTHONPATH="$src:$P529M_MODEL_SOURCE"
export OMP_NUM_THREADS=4
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="$((29000 + SLURM_JOB_ID % 10000))"
export root src out py manifest

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:4 \
  bash -lc '"$py" -m torch.distributed.run --nnodes=2 --nproc_per_node=4 --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" "$src/evaluate.py" --validation-manifest "$manifest" --expected-validation-sha256 6b53a58bbfe8991ecbbd754e8e27c3d832410f353a69f44a6180eb1e78624ae2 --validation-tokens 16678912 --microbatch 2 --checkpoint-id base --checkpoint "$root/formal/run-v5/resume.pt" --checkpoint-sha256 13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf --checkpoint-id f3_seed20260914 --checkpoint "$root/posttrain/screens-v2/F3_broad_multilingual_no_synthetic-1588454/train/model-final.pt" --checkpoint-sha256 4212f9aded17a8b5b0047efc83c7c2ba7d1198b51c2e4b63bc54e3daca6d7d4c --checkpoint-id f3_seed20260915 --checkpoint "$root/posttrain/screens-v3/F3_broad_multilingual_no_synthetic-seed20260915-1588560/train/model-final.pt" --checkpoint-sha256 dcc9a4f935d8b5574dc449b6263c696d9d355e70d157c8914f647c7aed61324a --output "$out/result.json"'

sha256sum "$out/result.json"
