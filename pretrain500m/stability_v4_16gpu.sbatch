#!/usr/bin/env bash
#SBATCH --job-name=p529m-stability-v4
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/stability-v4-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/stability-v4-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
data="$root/data/fineweb-edu-decontam/web/train-npy"
out="$root/pilots/stability-v4-$SLURM_JOB_ID"
mkdir -p "$out"
export root py PYTHONPATH="$root/source/train-v2" OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Ring NCCL_PROTO=Simple
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="$((26000 + SLURM_JOB_ID % 14000))"
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:4 \
  bash -lc 'export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-stability-v4-'"$SLURM_JOB_ID"'-${SLURM_NODEID}-inductor"; export TRITON_CACHE_DIR="/tmp/p529m-stability-v4-'"$SLURM_JOB_ID"'-${SLURM_NODEID}-triton"; "$py" -m torch.distributed.run --nnodes=4 --nproc_per_node=4 --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" "$root/source/train-v2/train_v3.py" --data-dir '"$data"' --val-dir '"$data"' --output-dir '"$out"' --architecture deep --target-tokens 1000000000 --max-data-epochs 100 --microbatch 4 --accumulation 16 --peak-lr 0.0015 --warmup-tokens 104857600 --save-every 999 --validate-every 100 --dense-bf16-tflops-per-gpu 209.5 --min-mfu 0.65 --mfu-grace-steps 5 --mfu-window 10 --pilot-steps 100 --no-save-final --deterministic --ddp-gradient-reduction bf16 --ddp-bucket-cap-mb 200 --protocol-file '"$root/source/protocol-v4.json" \
  > "$root/logs/stability-v4-$SLURM_JOB_ID-train.log" 2>&1
"$py" - "$out" <<'PY'
import hashlib,json,statistics,sys,time
from pathlib import Path
out=Path(sys.argv[1]); rows=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
train=[x for x in rows if 'mfu' in x]; status=json.loads((out/'training-status.json').read_text()); manifest=json.loads((out/'run-manifest.json').read_text())
steady=[float(x['mfu']) for x in train if int(x['step'])>5]
rolling=[statistics.median(steady[i-9:i+1]) for i in range(9,len(steady))]
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
passed=(status.get('status')=='COMPLETED' and status.get('step')==100 and len(train)==100 and min(rolling)>0.65 and statistics.median(steady)>0.65)
r={'status':'PASS_100_STEP_STABILITY_MFU_STRICTLY_GREATER_THAN_0.65' if passed else 'FAIL_100_STEP_STABILITY',
   'steps':len(train),'steady_samples':len(steady),'rolling_windows':len(rolling),
   'median_mfu':statistics.median(steady),'minimum_step_mfu':min(steady),'minimum_rolling_median_mfu':min(rolling),
   'median_tokens_per_second':statistics.median(float(x['tokens_per_second']) for x in train if int(x['step'])>5),
   'final_loss':float(train[-1]['loss']),'final_validation_loss':float([x for x in rows if 'val_loss' in x][-1]['val_loss']),
   'train_sha256':manifest['code_sha256']['train.py'],'protocol_sha256':manifest['protocol_sha256'],
   'metrics_sha256':digest(out/'metrics.jsonl'),'manifest_sha256':digest(out/'run-manifest.json'),'unix':time.time()}
(out/'stability-receipt.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps(r,sort_keys=True))
if not passed: raise SystemExit(65)
PY
