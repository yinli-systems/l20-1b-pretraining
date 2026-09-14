#!/usr/bin/env bash
set -euo pipefail

ROOT=/ssd/scxi253/pretrain500m-20260912-v1
PY=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
PIPELINE_PID=${PIPELINE_PID:?PIPELINE_PID is required}
PIPELINE_RECEIPT="$ROOT/receipts/pipeline-dclm-topup-v1-complete.json"

while [[ ! -f "$PIPELINE_RECEIPT" ]]; do
  if ! kill -0 "$PIPELINE_PID" 2>/dev/null; then
    echo 'DCLM top-up pipeline ended without a completion receipt' >&2
    exit 40
  fi
  sleep 5
done

cd "$ROOT/source/cpt-confirmation-v1"
sha256sum -c SHA256SUMS
PYTHONPATH="$ROOT/source/cpt-confirmation-v1:$ROOT/source/train-v2" "$PY" -m pytest -q test_v2.py
cd "$ROOT/source/confirmation-runner-v1"
sha256sum -c SHA256SUMS
bash -n run.sbatch
cd "$ROOT/source/confirmation-manifests-dclm-topup-v1"
sha256sum -c SHA256SUMS
PYTHONPATH="$ROOT/source/confirmation-manifests-dclm-topup-v1" "$PY" -m pytest -q test_build.py

"$PY" "$ROOT/source/confirmation-manifests-dclm-topup-v1/build.py" \
  --root "$ROOT" --output "$ROOT/source/confirmation-inputs-expansion-v1" \
  >"$ROOT/logs/confirmation-input-build-dclm-topup-v1.log" 2>&1

"$PY" "$ROOT/source/launch-confirmation-v1/submit.py" \
  >"$ROOT/logs/confirmation-submit-v1.log" 2>&1
