#!/usr/bin/env bash
set -uo pipefail

root=/ssd/scxi253/pretrain500m-20260912-v1
src="$root/source/long-confirmation-eval-v1"
py=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
plan="$src/checkpoint-plan.json"
receipt="$root/receipts/long-confirmation-eval-v1-submission.json"
log="$root/logs/long-confirmation-eval-v1-watcher.log"
protocol="$root/source/confirmation-inputs-expansion-v1/confirmation-protocol.json"
inputs="$root/source/confirmation-inputs-expansion-v1/build-receipt.json"
protocol_sha=d0c0fbb59bda58de814ab9f297ff4c8f00e9349b8e3f565ce0bb7e0fea4873f5
inputs_sha=1f2ee8a9a28c886877886fd34bb56f39beb46938047adbb1d9df4150715c71f5
export P529M_MODEL_SOURCE="$root/source/train-v2"
export PYTHONPATH="$src:$P529M_MODEL_SOURCE"

exec 9>"$root/receipts/long-confirmation-eval-v1-watcher.lock"
flock -n 9 || exit 73
cd "$src"
sha256sum -c SHA256SUMS >>"$log" 2>&1 || exit 74

while true; do
  if [ -f "$receipt" ]; then
    exit 0
  fi
  if [ ! -f "$plan" ]; then
    "$py" "$src/build_plan.py" --root "$root" \
      --protocol "$protocol" --expected-protocol-sha256 "$protocol_sha" \
      --inputs-receipt "$inputs" --expected-inputs-receipt-sha256 "$inputs_sha" \
      --output "$plan" >>"$log" 2>&1 || {
        sleep 30
        continue
      }
  fi
  plan_sha=$(sha256sum "$plan"); plan_sha=${plan_sha%% *}
  submit=$(sbatch --parsable --exclude=wqd10nbl11g5 \
    --export=ALL,EXPECTED_CHECKPOINT_PLAN_SHA256="$plan_sha" "$src/run.sbatch" 2>>"$log") || {
      sleep 30
      continue
    }
  job_id=${submit%%;*}
  export job_id plan_sha receipt
  "$py" - <<'PY'
import json
import os
import tempfile
import time

path = os.environ['receipt']
row = {'schema': 'p529m-long-confirmation-eval-submission-v1',
       'status': 'SUBMITTED_PENDING_ALLOCATION',
       'checked_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
       'job_id': int(os.environ['job_id']),
       'checkpoint_plan': '/ssd/scxi253/pretrain500m-20260912-v1/source/long-confirmation-eval-v1/checkpoint-plan.json',
       'checkpoint_plan_sha256': os.environ['plan_sha'],
       'claim_boundary': 'evaluation submitted; allocation, completion, selection and promotion remain unverified'}
fd, temp = tempfile.mkstemp(prefix='.long-confirm-eval-submit.', dir=os.path.dirname(path))
with os.fdopen(fd, 'w') as handle:
    json.dump(row, handle, indent=2, sort_keys=True)
    handle.write('\n')
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temp, path)
PY
  exit 0
done
