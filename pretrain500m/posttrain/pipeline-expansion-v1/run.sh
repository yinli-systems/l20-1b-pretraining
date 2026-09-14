#!/usr/bin/env bash
set -euo pipefail

ROOT=/ssd/scxi253/pretrain500m-20260912-v1
PY=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
LOG_ROOT="$ROOT/logs/pipeline-expansion-v1"
mkdir -p "$LOG_ROOT"

inputs=(
  "$ROOT/data/diverse-intake-v1"
  "$ROOT/data/diverse-intake-v2"
  "$ROOT/data/diverse-intake-v3"
  "$ROOT/data/diverse-intake-v4-finemath"
  "$ROOT/data/intake-expansion-v1"
)
input_args=()
for path in "${inputs[@]}"; do
  input_args+=(--input "$path")
done

while kill -0 4167022 2>/dev/null; do
  sleep 5
done
if pgrep -f "$ROOT/source/intake-expansion-v1/intake.py" >/dev/null; then
  echo 'intake process still exists' >&2
  exit 20
fi
"$PY" - <<'PY'
import json
import os
from pathlib import Path
root = Path('/ssd/scxi253/pretrain500m-20260912-v1/data/intake-expansion-v1')
summary = json.loads((root / 'intake-summary.json').read_text())
assert summary['status'] == 'RAW_INTAKE_FINISHED_NOT_ADMITTED'
assert summary['sources_ready'] == 16 and summary['sources_blocked'] == 0
assert len(list(root.glob('*.jsonl.gz'))) == 16
assert len(list(root.glob('*.receipt.json'))) == 16
assert not list(root.glob('*.part'))
lock = root / 'writer.lock'
if lock.exists():
    os.unlink(lock)
PY

for output in \
  "$ROOT/data/diverse-audit-expansion-v1" \
  "$ROOT/data/contamination-expansion-v1" \
  "$ROOT/data/legacy-screen-expansion-v1" \
  "$ROOT/data/old-source-overlap-expansion-v1" \
  "$ROOT/data/quality-family-expansion-v1" \
  "$ROOT/data/family-split-expansion-v1" \
  "$ROOT/data/selected-packs-expansion-v1"; do
  if [[ -e "$output" ]]; then
    echo "refusing existing output: $output" >&2
    exit 21
  fi
done
if [[ -e "$ROOT/receipts/exclusion-union-expansion-v1.json" ]]; then
  echo 'refusing existing exclusion output' >&2
  exit 22
fi

"$PY" "$ROOT/source/raw-audit-expansion-v1/audit_incremental.py" \
  "${input_args[@]}" \
  --prior-report "$ROOT/data/diverse-audit-v2-four-tranche-r1/report.json" \
  --tokenizer "$ROOT/formal/hf-v5/tokenizer.json" \
  --expected-tokenizer-sha256 30d71356c5ba154006df5bbb4a0583fc434525ceeb2f27a7d8a237ce5db26dc6 \
  --plan "$ROOT/source/intake-expansion-v1/plan.json" \
  --output "$ROOT/data/diverse-audit-expansion-v1" \
  --workers 4 \
  >"$LOG_ROOT/raw-audit.log" 2>&1 &
raw_pid=$!

PYTHONPATH="$ROOT/overlay-contamination-v1" \
"$PY" "$ROOT/source/contamination-expansion-v1/scan.py" \
  "${input_args[@]}" \
  --scan-from-tranche 4 \
  --bundle "$ROOT/evaluation/benchmark-reservation-v1/data" \
  --output "$ROOT/data/contamination-expansion-v1" \
  >"$LOG_ROOT/contamination.log" 2>&1 &
contamination_pid=$!

"$PY" "$ROOT/source/legacy-screen-expansion-v1/scan.py" \
  "${input_args[@]}" \
  --scan-from-tranche 4 \
  --database "$ROOT/data/decontam-13word.sqlite" \
  --expected-database-sha256 661f939a8787d2c14a52b305ec6a6b5737aabe5146f557461473e89f22e3a256 \
  --audit-source "$ROOT/source/raw-audit-expansion-v1" \
  --output "$ROOT/data/legacy-screen-expansion-v1" \
  --workers 4 \
  >"$LOG_ROOT/legacy.log" 2>&1 &
legacy_pid=$!

parallel_failed=0
wait "$raw_pid" || parallel_failed=1
wait "$contamination_pid" || parallel_failed=1
wait "$legacy_pid" || parallel_failed=1
if [[ "$parallel_failed" != 0 ]]; then
  echo 'parallel audit stage failed' >&2
  exit 30
fi

legacy_report="$ROOT/data/legacy-screen-expansion-v1/report.json"
legacy_sha=$(sha256sum "$legacy_report" | cut -d' ' -f1)
PYTHONPATH="$ROOT/overlay" "$PY" "$ROOT/source/old-source-overlap-expansion-v1/scan.py" \
  --manifest "$ROOT/source/old-source-overlap-expansion-v1/old-source-manifest.json" \
  --expected-manifest-sha256 8291e6c2a9af017904a3cc17b838cff231ff1a6da4cf7b9fe810b2dd6995a6e5 \
  --index-report "$legacy_report" \
  --expected-index-report-sha256 "$legacy_sha" \
  --output "$ROOT/data/old-source-overlap-expansion-v1" \
  --workers 16 \
  >"$LOG_ROOT/old-source.log" 2>&1

"$PY" "$ROOT/source/exclusion-union-expansion-v1/build.py" \
  --prior "$ROOT/receipts/exclusion-union-v2-four-tranche.json" \
  --supplemental-report "$ROOT/data/contamination-expansion-v1/report.json" \
  --legacy-report "$legacy_report" \
  --old-source-report "$ROOT/data/old-source-overlap-expansion-v1/report.json" \
  --output "$ROOT/receipts/exclusion-union-expansion-v1.json" \
  >"$LOG_ROOT/exclusion-union.log" 2>&1

raw_report="$ROOT/data/diverse-audit-expansion-v1/report.json"
raw_sha=$(sha256sum "$raw_report" | cut -d' ' -f1)
exclusions="$ROOT/receipts/exclusion-union-expansion-v1.json"
exclusions_sha=$(sha256sum "$exclusions" | cut -d' ' -f1)
PYTHONPATH="$ROOT/overlay-quality-family-v1" \
"$PY" "$ROOT/source/quality-family-expansion-v1/scan.py" \
  --raw-report "$raw_report" \
  --expected-raw-report-sha256 "$raw_sha" \
  --exclusions "$exclusions" \
  --expected-exclusions-sha256 "$exclusions_sha" \
  --audit-source "$ROOT/source/raw-audit-expansion-v1" \
  --output "$ROOT/data/quality-family-expansion-v1" \
  --workers 16 \
  >"$LOG_ROOT/quality.log" 2>&1

features="$ROOT/data/quality-family-expansion-v1/report.json"
features_sha=$(sha256sum "$features" | cut -d' ' -f1)
PYTHONPATH="$ROOT/overlay-quality-family-v1" \
"$PY" "$ROOT/source/family-split-expansion-v1/scan.py" \
  --features-report "$features" \
  --expected-features-sha256 "$features_sha" \
  --raw-report "$raw_report" \
  --exclusions "$exclusions" \
  --design "$ROOT/source/fast-start-nosynthetic-v1/experiment-design.json" \
  --expected-design-sha256 3d58055a1d2398b90c963c9ed4be7e99e1d1644281b95599c4ad5339d805d105 \
  --output "$ROOT/data/family-split-expansion-v1" \
  --workers 4 \
  >"$LOG_ROOT/family.log" 2>&1

family="$ROOT/data/family-split-expansion-v1/report.json"
family_sha=$(sha256sum "$family" | cut -d' ' -f1)
"$PY" "$ROOT/source/pack-selected-expansion-v1/pack.py" \
  --family-report "$family" \
  --expected-family-sha256 "$family_sha" \
  --tokenizer "$ROOT/formal/hf-v5/tokenizer.json" \
  --expected-tokenizer-sha256 30d71356c5ba154006df5bbb4a0583fc434525ceeb2f27a7d8a237ce5db26dc6 \
  --output "$ROOT/data/selected-packs-expansion-v1" \
  --workers 4 \
  >"$LOG_ROOT/pack.log" 2>&1

"$PY" - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
root = Path('/ssd/scxi253/pretrain500m-20260912-v1')
artifacts = [
    root/'data/diverse-audit-expansion-v1/report.json',
    root/'data/contamination-expansion-v1/report.json',
    root/'data/legacy-screen-expansion-v1/report.json',
    root/'data/old-source-overlap-expansion-v1/report.json',
    root/'receipts/exclusion-union-expansion-v1.json',
    root/'data/quality-family-expansion-v1/report.json',
    root/'data/family-split-expansion-v1/report.json',
    root/'data/selected-packs-expansion-v1/report.json',
]
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
receipt = {
    'status': 'EXPANSION_FILTER_AND_PACK_COMPLETE_NOT_TRAINING_ADMITTED',
    'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'artifacts': {str(p): digest(p) for p in artifacts},
    'training_admitted': False,
    'training_launched': False,
}
path = root/'receipts/pipeline-expansion-v1-complete.json'
path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
PY
