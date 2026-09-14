#!/usr/bin/env bash
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:30:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.err
set -euo pipefail

root=/ssd/scxi253/pretrain500m-20260912-v1
src="$root/source/cpt-confirmation-v1"
inputs="$root/source/confirmation-inputs-expansion-v1"
recipe=${RECIPE:?RECIPE must name one frozen confirmation recipe}
seed=${RUN_SEED:?RUN_SEED must name one frozen confirmation seed}
manifest="$inputs/manifests/$recipe.json"
admission="$inputs/admissions/$recipe.admission.json"
validation="$root/data/development-masked-v2/development-mixture.json"
protocol="$inputs/confirmation-protocol.json"
initial="$root/formal/run-v5/resume.pt"
out="$root/posttrain/confirmations-v1/$recipe-seed$seed-fragmented4x1-$SLURM_JOB_ID"
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python

test "$SLURM_NNODES" -eq 4
test -f "$manifest"
test -f "$admission"
test ! -e "$out"
mkdir -p "$out"
cd "$src"
sha256sum -c SHA256SUMS

export out
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:1 \
  bash -lc 'set -euo pipefail; f="$out/gpu-${SLURM_NODEID}.csv"; nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv > "$f"; test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c "^NVIDIA GeForce RTX 5090$")" -eq 1'

python3 - "$out" <<'PY'
import csv
import glob
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
files = sorted(glob.glob(str(out / "gpu-*.csv")))
if len(files) != 4:
    raise SystemExit(f"expected four node GPU inventories, found {len(files)}")
rows = []
for path in files:
    with open(path, newline="") as handle:
        values = list(csv.DictReader(handle))
    if len(values) != 1 or values[0][" name"].strip() != "NVIDIA GeForce RTX 5090":
        raise SystemExit(f"invalid one-GPU inventory in {path}: {values}")
    rows.append(values[0])
uuids = {row[" uuid"].strip() for row in rows}
if len(uuids) != 4:
    raise SystemExit(f"GPU UUIDs are not distinct: {sorted(uuids)}")
PY

mixture_sha=$(sha256sum "$manifest"); mixture_sha=${mixture_sha%% *}
admission_sha=$(sha256sum "$admission"); admission_sha=${admission_sha%% *}
validation_sha=$(sha256sum "$validation"); validation_sha=${validation_sha%% *}

export P529M_MODEL_SOURCE="$root/source/train-v2"
export PYTHONPATH="$src:$P529M_MODEL_SOURCE"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=$((26000 + SLURM_JOB_ID % 10000))
export root src inputs recipe seed manifest admission validation protocol initial py mixture_sha admission_sha validation_sha

srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --cpus-per-task=6 --gres=gpu:1 \
  bash -lc 'export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-confirmation-fragmented-$SLURM_JOB_ID-${SLURM_NODEID}-inductor"; export TRITON_CACHE_DIR="/tmp/p529m-confirmation-fragmented-$SLURM_JOB_ID-${SLURM_NODEID}-triton"; "$py" -m torch.distributed.run --nnodes=4 --nproc_per_node=1 --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" "$src/train_cpt.py" --mixture-manifest "$manifest" --expected-mixture-sha256 "$mixture_sha" --validation-manifest "$validation" --expected-validation-sha256 "$validation_sha" --validation-tokens 16023552 --admission-receipt "$admission" --expected-admission-sha256 "$admission_sha" --protocol-file "$protocol" --output-dir "$out/train" --target-tokens 2147483648 --pilot-steps 1024 --architecture deep --microbatch 4 --accumulation 64 --peak-lr 0.0001 --warmup-tokens 16777216 --save-every 9999 --validate-every 512 --seed "$seed" --init-model-checkpoint "$initial" --expected-init-sha256 13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf --dense-bf16-tflops-per-gpu 209.5 --min-mfu 0.50 --mfu-grace-steps 5 --mfu-window 10 --deterministic --validate-before-training --checkpoint-mode model-only-final'
