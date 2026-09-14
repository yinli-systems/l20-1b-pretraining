#!/usr/bin/env bash
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --job-name=p529m-cpt16-safe
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.err
set -euo pipefail

root=/ssd/scxi253/pretrain500m-20260912-v1
src="$root/source/cpt-v8"
inputs="$root/source/screen-inputs-v2"
recipe=F3_broad_multilingual_no_synthetic
seed=20260915
manifest="$inputs/manifests-v2/$recipe.json"
admission="$inputs/admission-v1/$recipe.admission.json"
validation="$root/data/development-masked-v2/development-mixture.json"
protocol="$inputs/cpt-pilot-protocol-v1.json"
initial="$root/formal/run-v5/resume.pt"
out="$root/posttrain/scale-cpt-v8-16gpu-v1/$SLURM_JOB_ID"
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python

test "$SLURM_NNODES" -eq 4
test -f "$manifest"
test -f "$admission"
test ! -e "$out"
mkdir -p "$out"
cd "$src"
sha256sum -c SHA256SUMS

srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:4 \
  bash -lc 'printf "HOST=%s\n" "$(hostname)"; nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader' \
  > "$out/gpu-inventory.txt"

mixture_sha=$(sha256sum "$manifest"); mixture_sha=${mixture_sha%% *}
admission_sha=$(sha256sum "$admission"); admission_sha=${admission_sha%% *}
validation_sha=$(sha256sum "$validation"); validation_sha=${validation_sha%% *}
world=16
accumulation=16

export root src manifest admission validation protocol initial out py
export mixture_sha admission_sha validation_sha world accumulation seed
export P529M_MODEL_SOURCE="$root/source/train-v2"
export PYTHONPATH="$src:$P529M_MODEL_SOURCE"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="$((28000 + SLURM_JOB_ID % 10000))"

srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:4 \
  bash -lc 'export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-cpt8-'"$SLURM_JOB_ID"'-${SLURM_NODEID}-inductor"; export TRITON_CACHE_DIR="/tmp/p529m-cpt8-'"$SLURM_JOB_ID"'-${SLURM_NODEID}-triton"; "$py" -m torch.distributed.run --nnodes=4 --nproc_per_node=4 --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" "$src/train_cpt.py" --mixture-manifest "$manifest" --expected-mixture-sha256 "$mixture_sha" --validation-manifest "$validation" --expected-validation-sha256 "$validation_sha" --validation-tokens 16023552 --validation-microbatch 1 --admission-receipt "$admission" --expected-admission-sha256 "$admission_sha" --protocol-file "$protocol" --output-dir "$out/train" --target-tokens 536870912 --pilot-steps 20 --architecture deep --microbatch 4 --accumulation "$accumulation" --peak-lr 0.0001 --warmup-tokens 16777216 --save-every 999 --validate-every 256 --seed "$seed" --init-model-checkpoint "$initial" --expected-init-sha256 13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf --dense-bf16-tflops-per-gpu 209.5 --min-mfu 0.50 --mfu-grace-steps 5 --mfu-window 10 --deterministic --checkpoint-mode none'

"$py" - "$out" <<'PY'
from pathlib import Path
import hashlib,json,math,os,statistics,sys,time
out=Path(sys.argv[1]); train=out/'train'
rows=[]
for line in (train/'metrics.jsonl').read_text().splitlines():
    try: rows.append(json.loads(line))
    except: pass
steps=[r for r in rows if isinstance(r.get('mfu'),(int,float))]
vals=[float(r['mfu']) for r in steps if r['step']>5]
validations=[r for r in rows if isinstance(r.get('val_loss_equal_domain'),(int,float))]
status=json.loads((train/'training-status.json').read_text())
manifest=json.loads((train/'run-manifest.json').read_text())
gpu_lines=[line for line in (out/'gpu-inventory.txt').read_text().splitlines() if 'NVIDIA GeForce RTX 5090' in line]
uuids=[line.rsplit(', ',1)[-1] for line in gpu_lines]
finite=all(math.isfinite(float(r[k])) for r in steps for k in ('loss','grad_norm','mfu','step_seconds'))
median=statistics.median(vals)
final10=statistics.median(vals[-10:])
passed=(len(steps)==20 and len(vals)==15 and final10>0.50 and finite and
        len(gpu_lines)==16 and len(set(uuids))==16 and validations and
        status.get('status')=='QUALIFICATION_COMPLETED' and status.get('step')==20 and
        manifest.get('world_size')==16 and manifest.get('validation_microbatch')==1 and
        manifest.get('ddp_policy')=={'find_unused_parameters':True,'static_graph':False,
                                    'gradient_as_bucket_view':True,'bucket_cap_mb':25} and
        all(not r.get('ddp_buckets_rebuilt') for r in steps))
result={
 'schema_version':1,'status':'PASS_MFU_AND_INTEGRITY' if passed else 'FAIL',
 'world_size':16,'nodes':4,'gpus_per_node':4,'gpu_model':'NVIDIA GeForce RTX 5090',
 'distinct_gpu_uuids':len(set(uuids)),'microbatch':4,'accumulation':32,
 'validation_microbatch':1,'global_prediction_tokens_per_step':2097152,
 'steps':len(steps),'steady_steps':'6..20','steady_mfu_median':median,
 'final_mfu10_median':final10,'median_tokens_per_second':statistics.median(float(r['tokens_per_second']) for r in steps[5:]),
 'finite_training_metrics':finite,'val_loss_equal_domain':validations[-1]['val_loss_equal_domain'] if validations else None,
 'run_fingerprint_sha256':manifest.get('run_fingerprint_sha256'),
 'claim_boundary':'multi-node throughput qualification only; no quality or promotion claim',
 'created_unix':time.time(),
}
p=out/'scale-receipt.json'; tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); os.replace(tmp,p)
print(json.dumps(result,indent=2,sort_keys=True))
print('RECEIPT_SHA256',hashlib.sha256(p.read_bytes()).hexdigest())
if not passed: raise SystemExit(66)
PY
