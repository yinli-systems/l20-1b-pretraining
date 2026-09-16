#!/usr/bin/env bash
#SBATCH --job-name=p529m-f2-export-newseeds
#SBATCH --partition=gpu_5090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:45:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/%x-%j.err
set -euo pipefail

root=/ssd/scxi253/pretrain500m-20260912-v1
src="$root/source/f2-export-parity-independent-samples-v1"
plan="$src/plan.json"
out="$root/posttrain/f2-export-parity-independent-samples-v1/$SLURM_JOB_ID"
tmp_root="/tmp/p529m-f2-export-newseeds-$SLURM_JOB_ID"
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
tokenizer="$root/source/pack-v3/tokenizer"
test ! -e "$out"
test ! -e "$tmp_root"
mkdir -p "$out" "$tmp_root"
cleanup() { rm -rf -- "$tmp_root"; }
trap cleanup EXIT
cd "$src"
sha256sum -c SHA256SUMS
echo '44bc8f5a49e7d4a6d55d18a5f3e9af3ea6cb0ce09cc4822342ebf35ed3f39c5e  plan.json' | sha256sum -c -
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA GeForce RTX 5090$')" -eq 1
test "$(df -B1 --output=avail "$tmp_root" | tail -n 1)" -gt 4294967296
test "$(df -B1 --output=avail "$out" | tail -n 1)" -gt 2147483648
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv > "$out/gpu.csv"
export PYTHONPATH="$root/source/train-v2"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mapfile -t checkpoints < <("$py" - "$plan" <<'PY'
import json,sys
plan=json.load(open(sys.argv[1]))
assert plan['schema']=='p529m-f2-export-parity-independent-samples-plan-v1'
assert plan['status']=='FROZEN_BEFORE_EXECUTION'
assert plan['sample_seeds']==[20260916,20260917] and plan['sample_shape']==[2,128]
assert plan['original_functional_limits']=={'max_abs_logit_difference':0.5,'mean_abs_logit_difference':0.025,'minimum_argmax_agreement':0.97}
assert [x['seed'] for x in plan['checkpoint_grid']]==[20260914,20260915]
for item in plan['checkpoint_grid']:
    assert item['id']==f"continuation_seed{item['seed']}" and item['expected_step']==256
    print('\t'.join(str(item[k]) for k in ('id','checkpoint','checkpoint_sha256','expected_step')))
PY
)
test "${#checkpoints[@]}" -eq 2
overall_rc=0
for sample_seed in 20260916 20260917; do
  for row in "${checkpoints[@]}"; do
    IFS=$'\t' read -r candidate_id checkpoint checkpoint_sha expected_step <<< "$row"
    echo "$checkpoint_sha  $checkpoint" | sha256sum -c -
    cell_id="sample${sample_seed}-${candidate_id}"
    mkdir -p "$out/$cell_id"
    if ! "$py" "$src/diagnose_export.py" \
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
    fi
    rm -rf -- "$tmp_root/$cell_id"
  done
done
"$py" "$src/summarize.py" --plan "$plan" --output-root "$out" --job-id "$SLURM_JOB_ID"
sha256sum "$out/result.json"
exit "$overall_rc"
