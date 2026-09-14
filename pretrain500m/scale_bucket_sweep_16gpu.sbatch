#!/usr/bin/env bash
#SBATCH --job-name=p529m-bucket-sweep
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/bucket-sweep-%A-%a.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/bucket-sweep-%A-%a.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
buckets=(100 200 400); bucket=${buckets[$SLURM_ARRAY_TASK_ID]}
world=$((SLURM_NNODES * 4))
accumulation=$((256 / world))
out="$root/pilots/bucket-sweep-$SLURM_ARRAY_JOB_ID/task-$SLURM_ARRAY_TASK_ID-bucket-$bucket"
data="$root/data/fineweb-edu-decontam/web/train-npy"
mkdir -p "$out"
export root py
export PYTHONPATH="$root/source/train-v2"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="$((26000 + (SLURM_ARRAY_JOB_ID + SLURM_ARRAY_TASK_ID) % 14000))"
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:4 \
  bash -lc 'export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-bucket-'"$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"'-${SLURM_NODEID}-inductor"; export TRITON_CACHE_DIR="/tmp/p529m-bucket-'"$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"'-${SLURM_NODEID}-triton"; "$py" -m torch.distributed.run --nnodes='"$SLURM_NNODES"' --nproc_per_node=4 --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" "$root/source/train-v2/train_v3.py" --data-dir '"$data"' --val-dir '"$data"' --output-dir '"$out"' --architecture deep --target-tokens 1000000000 --max-data-epochs 100 --microbatch 4 --accumulation '"$accumulation"' --peak-lr 0.0015 --warmup-tokens 104857600 --save-every 999 --validate-every 12 --dense-bf16-tflops-per-gpu 209.5 --min-mfu 0.0 --mfu-grace-steps 2 --mfu-window 4 --pilot-steps 12 --no-save-final --deterministic --ddp-gradient-reduction bf16 --ddp-bucket-cap-mb '"$bucket"' --protocol-file '"$root/source/protocol-v2.json" \
  > "$root/logs/bucket-sweep-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID-train.log" 2>&1
"$py" - "$out/metrics.jsonl" "$out/run-manifest.json" "$out/scale-receipt.json" "$bucket" <<'PY'
import hashlib,json,statistics,sys,time
metrics,manifest_path,receipt_path,bucket=sys.argv[1:]
rows=[json.loads(line) for line in open(metrics) if 'mfu' in line]
steady=rows[2:]
manifest=json.load(open(manifest_path))
digest=lambda p:hashlib.sha256(open(p,'rb').read()).hexdigest()
receipt={
  'status':'PASS_MFU_STRICTLY_GREATER_THAN_0.65' if statistics.median(r['mfu'] for r in steady)>0.65 else 'FAIL_MFU_NOT_GREATER_THAN_0.65',
  'nodes':4,'gpus':16,'microbatch':4,'accumulation':16,
  'global_prediction_tokens_per_step':2097152,'samples':len(steady),
  'ddp_gradient_reduction':'bf16','ddp_bucket_cap_mb':float(bucket),
  'nccl_algo':'Ring','nccl_proto':'Simple',
  'median_mfu':statistics.median(r['mfu'] for r in steady),
  'median_tokens_per_second':statistics.median(r['tokens_per_second'] for r in steady),
  'minimum_mfu':min(r['mfu'] for r in steady),'maximum_mfu':max(r['mfu'] for r in steady),
  'train_sha256':manifest['code_sha256']['train.py'],'protocol_sha256':manifest['protocol_sha256'],
  'metrics_sha256':digest(metrics),'manifest_sha256':digest(manifest_path),'unix':time.time(),
}
open(receipt_path,'w').write(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,sort_keys=True))
if receipt['status'].startswith('FAIL'):raise SystemExit(65)
PY
