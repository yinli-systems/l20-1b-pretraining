#!/usr/bin/env bash
#SBATCH --job-name=p500m-dispatch-dedup-v1
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:20:00
#SBATCH --output=/ssd/scxi253/pretrain500m-20260912-v1/logs/dispatch-dedup-v1-%j.out
#SBATCH --error=/ssd/scxi253/pretrain500m-20260912-v1/logs/dispatch-dedup-v1-%j.err
set -euo pipefail
root=/ssd/scxi253/pretrain500m-20260912-v1
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
receipt="${P500M_DEDUP_RECEIPT:-$root/data/fineweb-edu-audit-v1/cross-part-dedup.json}"
audit_dir="$(dirname "$receipt")"
status=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$receipt")
pack_job=
audit_job=
if [[ "$status" == PASS_ZERO_CROSS_PART_DUPLICATES ]]; then
  parts="$root/data/fineweb-edu-parts"
  final_receipt="$receipt"
elif [[ "$status" == RERUN_WITH_EXCLUSIONS_REQUIRED ]]; then
  plan="$audit_dir/rerun-plan.json"
  "$py" "$root/source/pack-v3/prepare_dedup_rerun.py" \
    --parts-root "$root/data/fineweb-edu-parts" \
    --audit-receipt "$receipt" \
    --output-root "$root/data/fineweb-edu-parts-dedup-v2" \
    --plan "$plan"
  array_spec=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["slurm_array_spec"])' "$plan")
  if [[ -z "$array_spec" ]]; then
    printf 'audit requires rerun but plan has no task indices\n' >&2
    exit 67
  fi
  pack_job=$(sbatch --array="$array_spec" \
    --export=ALL,P500M_EXCLUDE_ROOT="$audit_dir" \
    "$root/source/pack-v3/pack_array_dedup.sbatch" | awk '{print $4}')
  audit_job=$(sbatch --dependency="afterok:$pack_job" "$root/source/pack-v3/audit_partitions_v2.sbatch" | awk '{print $4}')
  parts="$root/data/fineweb-edu-parts-dedup-v2"
  final_receipt="$root/data/fineweb-edu-audit-v2/cross-part-dedup.json"
else
  printf 'unexpected audit status: %s\n' "$status" >&2
  exit 68
fi

dependency=()
if [[ -n "$audit_job" ]]; then dependency=(--dependency="afterok:$audit_job"); fi
finalize_job=$(sbatch "${dependency[@]}" \
  --export=ALL,P500M_PARTS_ROOT="$parts",P500M_DEDUP_RECEIPT="$final_receipt" \
  "$root/source/pack-v3/finalize_packed_data.sbatch" | awk '{print $4}')
selector_state=$(sacct -X -j 1584117 --noheader --parsable2 --format=State \
  | awk -F'|' 'NF && $1 != "" {print $1; exit}')
formal_dependency="afterok:$finalize_job"
case "$selector_state" in
  COMPLETED*) ;;
  PENDING*|RUNNING*|CONFIGURING*|COMPLETING*)
    formal_dependency+=":1584117"
    ;;
  *)
    printf 'LR selector 1584117 is not usable: %s\n' "$selector_state" >&2
    exit 69
    ;;
esac
formal_dispatch_job=$(sbatch --dependency="$formal_dependency" \
  "$root/source/train-v2/dispatch_formal.sbatch" | awk '{print $4}')
"$py" - "$root" "$status" "$pack_job" "$audit_job" "$finalize_job" "$formal_dispatch_job" "$parts" "$final_receipt" <<'PY'
import json,os,sys,time
root,status,pack_job,audit_job,finalize_job,formal_dispatch_job,parts,receipt=sys.argv[1:]
value={
  'status':'DISPATCHED',
  'source_audit_status':status,
  'dedup_pack_job':pack_job or None,
  'dedup_audit_job':audit_job or None,
  'finalize_job':finalize_job,
  'formal_dispatch_job':formal_dispatch_job,
  'final_parts_root':parts,
  'final_dedup_receipt':receipt,
  'unix':time.time(),
}
path=os.path.join(root,'receipts',f'dedup-dispatch-{os.environ["SLURM_JOB_ID"]}.json')
tmp=path+'.tmp';open(tmp,'w').write(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)
print(json.dumps(value,sort_keys=True))
PY
