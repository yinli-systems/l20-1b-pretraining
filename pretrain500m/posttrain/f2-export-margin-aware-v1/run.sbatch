#!/usr/bin/env bash
#SBATCH --job-name=p529m-f2-margin-export
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%A_%a.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%A_%a.err
set -euo pipefail

root=/ssd/scxi253/pretrain500m-20260912-v1
src="$root/source/f2-export-margin-aware-v1"
plan="$src/plan.json"
out="$root/posttrain/f2-export-margin-aware-v1/$SLURM_ARRAY_JOB_ID/task-$SLURM_ARRAY_TASK_ID"
tmp_root="/tmp/p529m-f2-margin-export-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
tokenizer="$root/source/pack-v3/tokenizer"
test ! -e "$out"
test ! -e "$tmp_root"
mkdir -p "$out" "$tmp_root"
cleanup() { rm -rf -- "$tmp_root"; }
trap cleanup EXIT
cd "$src"
sha256sum -c SHA256SUMS
echo 'eb782b750ee95da76d78e5c285c9ba17f013dddba66ca977b6cd549dd153b9f6  plan.json' | sha256sum -c -
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA GeForce RTX 5090$')" -eq 1
test "$(df -B1 --output=avail "$tmp_root" | tail -n 1)" -gt 4294967296
test "$(df -B1 --output=avail "$out" | tail -n 1)" -gt 6442450944
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv > "$out/gpu.csv"
export PYTHONPATH="$root/source/train-v2"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mapfile -t checkpoints < <("$py" - "$plan" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
plan=json.load(open(sys.argv[1])); task=int(sys.argv[2])
assert plan['schema']=='p529m-f2-margin-aware-export-plan-v1'
assert plan['status']=='FROZEN_BEFORE_FRESH_SAMPLES'
assert plan['sample_seeds']==[20260918,20260919,20260920,20260921]
assert plan['sample_shape']==[2,128] and task in (0,1)
pair=plan['checkpoint_pairs'][task]
assert pair['pair_seed'] in (20260914,20260915)
assert [x['role'] for x in pair['checkpoints']]==['parent','continuation']
for item in pair['checkpoints']:
    print('\t'.join(str(item[k]) for k in ('id','role','checkpoint','checkpoint_sha256','expected_step')))
PY
)
test "${#checkpoints[@]}" -eq 2
for row in "${checkpoints[@]}"; do
  IFS=$'\t' read -r candidate_id role checkpoint checkpoint_sha expected_step <<< "$row"
  echo "$checkpoint_sha  $checkpoint" | sha256sum -c -
done

overall_rc=0
for sample_seed in 20260918 20260919 20260920 20260921; do
  for row in "${checkpoints[@]}"; do
    IFS=$'\t' read -r candidate_id role checkpoint checkpoint_sha expected_step <<< "$row"
    cell_id="sample${sample_seed}-${candidate_id}"
    mkdir -p "$out/$cell_id"
    if ! "$py" "$src/export_margin_aware.py" \
      --checkpoint "$checkpoint" --tokenizer-dir "$tokenizer" \
      --expected-step "$expected_step" --sample-seed "$sample_seed" \
      --output-dir "$tmp_root/$cell_id" \
      --diagnostic-output "$out/$cell_id/diagnostic.json"; then
      overall_rc=1
    fi
    if test -f "$out/$cell_id/diagnostic.json"; then
      sha256sum "$out/$cell_id/diagnostic.json"
    fi
    if test -f "$tmp_root/$cell_id/export-receipt.json"; then
      cp "$tmp_root/$cell_id/export-receipt.json" "$out/$cell_id/export-receipt.json"
      if test "$role" = continuation && test "$sample_seed" -eq 20260918; then
        cp -R "$tmp_root/$cell_id" "$out/$candidate_id-hf"
      fi
    fi
    rm -rf -- "$tmp_root/$cell_id"
  done
done
"$py" "$src/summarize_task.py" \
  --plan "$plan" --output-root "$out" \
  --array-job-id "$SLURM_ARRAY_JOB_ID" --array-task-id "$SLURM_ARRAY_TASK_ID"
sha256sum "$out/task-result.json"
exit "$overall_rc"
