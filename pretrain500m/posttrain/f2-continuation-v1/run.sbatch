#!/usr/bin/env bash
#SBATCH --job-name=f2cpt529
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.err
set -euo pipefail

root=/ssd/scxi253/pretrain500m-20260912-v1
source_dir="$root/source/cpt-confirmation-v1"
input_dir="$root/source/continuation-inputs-f2-v1/build-output"
manifest="$input_dir/F2_reasoning_no_synthetic.continuation.json"
admission="$input_dir/F2_reasoning_no_synthetic.continuation.admission.json"
protocol="$input_dir/f2-continuation-protocol.json"
validation="$root/data/development-masked-f2-continuation-v1/development-mixture.json"
receipt="$input_dir/build-receipt.json"
seed=${RUN_SEED:?RUN_SEED must be a frozen F2 seed}
case "$seed" in
  20260914)
    parent="$root/posttrain/confirmations-v1/F2_reasoning_no_synthetic-seed20260914-1589567/train/model-final.pt"
    parent_sha=69d9d8d3358f23815f2455f7e075645056f90fe01ed40a11525130a0e58ca085
    ;;
  20260915)
    parent="$root/posttrain/confirmations-v1/F2_reasoning_no_synthetic-seed20260915-1590004/train/model-final.pt"
    parent_sha=f6a4ff43aeeb892ce7dd3dd7ed69f022cef6af688f21b3f005f67850f6041cbc
    ;;
  *)
    echo "Invalid F2 seed" >&2
    exit 2
    ;;
esac

test -f "$manifest"
test -f "$admission"
test -f "$protocol"
test -f "$validation"
test -f "$receipt"
test -f "$parent"
out="$root/posttrain/f2-continuation-v1/seed${seed}-${SLURM_JOB_ID}"
test ! -e "$out"
mkdir -p "$out"
cd "$source_dir"
sha256sum -c SHA256SUMS
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv > "$out/gpu.csv"
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA GeForce RTX 5090$')
test "$gpu_count" -eq 4

"/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python" - "$manifest" "$admission" "$protocol" "$validation" "$receipt" "$parent" "$parent_sha" <<'PY'
from pathlib import Path
import hashlib,json,sys

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()

manifest,admission,protocol,validation,receipt,parent,parent_sha=map(Path,sys.argv[1:])
expected={'manifest':'c87df8089acf908280e6e974d1ae7a13b1faa02175ad531191f7cfab773e1819',
          'admission':'0d3ad34bea4c58f5566811675494fb68cab1ba91bf23a49c0336294ee9ffe8c9',
          'protocol':'0f4f7bd4a97fb3b4be4f9c095a012b33beb004af36c747fc7e2cb217b82a01f7',
          'validation':'b9791aff2b537689c165f504f3c0fb262952c9d87762e8c2aa3388c6b9e3aa3b'}
for name,path in zip(('manifest','admission','protocol','validation'),
                     (manifest,admission,protocol,validation)):
    assert sha(path)==expected[name],name
assert sha(parent)==str(parent_sha), 'parent checkpoint SHA'
admitted=json.loads(admission.read_text());built=json.loads(receipt.read_text())
assert admitted['status']=='PASS' and all(v=='PASS' for v in admitted['checks'].values())
assert built['status']=='PASS_F2_CONTINUATION_INPUTS_ADMITTED_NOT_TRAINED'
assert built['manifest_sha256']==expected['manifest'] and built['admission_sha256']==expected['admission']
assert admitted['training_manifest_sha256']==expected['manifest']
assert admitted['validation_manifest_sha256']==expected['validation']
assert admitted['protocol_sha256']==expected['protocol']
print('F2_CONTINUATION_JOB_INPUTS_PASS')
PY

py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
export P529M_MODEL_SOURCE="$root/source/train-v2"
export PYTHONPATH="$source_dir:$P529M_MODEL_SOURCE"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-f2cont-${SLURM_JOB_ID}-inductor"
export TRITON_CACHE_DIR="/tmp/p529m-f2cont-${SLURM_JOB_ID}-triton"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
port=$((26000 + SLURM_JOB_ID % 10000))

srun --nodes=1 --ntasks=1 --cpus-per-task=24 --gres=gpu:4 \
  "$py" -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port="$port" "$source_dir/train_cpt.py" \
  --mixture-manifest "$manifest" --expected-mixture-sha256 c87df8089acf908280e6e974d1ae7a13b1faa02175ad531191f7cfab773e1819 \
  --validation-manifest "$validation" --expected-validation-sha256 b9791aff2b537689c165f504f3c0fb262952c9d87762e8c2aa3388c6b9e3aa3b \
  --validation-tokens 18251776 \
  --admission-receipt "$admission" --expected-admission-sha256 0d3ad34bea4c58f5566811675494fb68cab1ba91bf23a49c0336294ee9ffe8c9 \
  --protocol-file "$protocol" \
  --output-dir "$out/train" --target-tokens 536870912 --pilot-steps 256 \
  --architecture deep --microbatch 4 --accumulation 64 \
  --peak-lr 0.00006 --warmup-tokens 16777216 \
  --save-every 128 --validate-every 128 --seed "$seed" \
  --init-model-checkpoint "$parent" --expected-init-sha256 "$parent_sha" \
  --dense-bf16-tflops-per-gpu 209.5 --min-mfu 0.70 \
  --mfu-grace-steps 5 --mfu-window 10 \
  --deterministic --validate-before-training --checkpoint-mode full
