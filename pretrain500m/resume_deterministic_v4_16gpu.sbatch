#!/usr/bin/env bash
#SBATCH --job-name=p529m-resume-det-v4
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:50:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/resume-det-v4-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/resume-det-v4-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
export root py PYTHONPATH="$root/source/train-v2" OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Ring NCCL_PROTO=Simple
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="$((28000 + SLURM_JOB_ID % 12000))"
data="$root/data/fineweb-edu-decontam/web/train-npy"
job_root="$root/pilots/resume-det-v4-$SLURM_JOB_ID"
base=(--data-dir "$data" --val-dir "$data" --architecture deep --target-tokens 100000000
      --microbatch 4 --accumulation 16 --peak-lr 0.0015 --warmup-tokens 104857600
      --save-every 2 --validate-every 4 --dense-bf16-tflops-per-gpu 209.5
      --min-mfu 0.0 --mfu-grace-steps 1 --mfu-window 2 --deterministic
      --ddp-gradient-reduction bf16 --ddp-bucket-cap-mb 200
      --protocol-file "$root/source/protocol-v4.json")
run_train() {
  local name=$1; shift
  srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:4 \
    bash -lc 'export TORCHINDUCTOR_CACHE_DIR="/tmp/p529m-det-v4-'"$SLURM_JOB_ID"'-${SLURM_NODEID}-inductor"; export TRITON_CACHE_DIR="/tmp/p529m-det-v4-'"$SLURM_JOB_ID"'-${SLURM_NODEID}-triton"; "$py" -m torch.distributed.run --nnodes=4 --nproc_per_node=4 --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" "$root/source/train-v2/train_v3.py" '"$*" \
    > "$root/logs/$name-$SLURM_JOB_ID.log" 2>&1
}
run_train det-v4-continuous "${base[@]}" --output-dir "$job_root/continuous" --pilot-steps 4
run_train det-v4-split-a "${base[@]}" --output-dir "$job_root/split" --pilot-steps 2
run_train det-v4-split-b "${base[@]}" --output-dir "$job_root/split" --pilot-steps 2 --resume
"$py" "$root/source/train-v2/compare_checkpoints.py" \
  "$job_root/continuous/resume.pt" "$job_root/split/resume.pt" \
  | tee "$job_root/resume-comparison.json"
"$py" - "$job_root" <<'PY'
import hashlib,json,os,sys,time
from pathlib import Path
root=Path(sys.argv[1]); paths=[root/'continuous'/'resume.pt',root/'split'/'resume.pt']
records=[]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(16*1024**2),b''):h.update(block)
 return h.hexdigest()
for p in paths:
 records.append({'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
for p in paths:p.unlink()
if any(p.exists() for p in paths):raise SystemExit('checkpoint cleanup verification failed')
r={'status':'PASS_EXACT_POST_COMPARISON_CLEANUP','removed':records,'verified_absent':True,'unix':time.time()}
(root/'checkpoint-cleanup.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps(r,sort_keys=True))
PY
printf 'DETERMINISTIC_RESUME_V4_PILOT_COMPLETED\n'
