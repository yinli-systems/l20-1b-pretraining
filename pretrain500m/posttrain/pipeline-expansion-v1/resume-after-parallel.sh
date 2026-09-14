#!/usr/bin/env bash
set -euo pipefail

ROOT=/ssd/scxi253/pretrain500m-20260912-v1
PY=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
LOG_ROOT="$ROOT/logs/pipeline-expansion-v1"

while kill -0 404720 2>/dev/null; do sleep 5; done
while kill -0 323506 2>/dev/null; do sleep 5; done

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path('/ssd/scxi253/pretrain500m-20260912-v1/data')
expected = {
    'diverse-audit-expansion-v1': 'COMBINED_RAW_INTAKE_MEASURED_NOT_ADMITTED',
    'contamination-expansion-v1': 'SUPPLEMENTAL_EXACT_SPAN_AUDIT_COMPLETE_ADMISSION_PENDING',
    'legacy-screen-expansion-v1': 'LEGACY_HASH_SCAN_COMPLETE_NOT_ADMITTED',
}
for name, status in expected.items():
    report = json.loads((root/name/'report.json').read_text())
    assert report['status'] == status, (name, report.get('status'))
PY

legacy_report="$ROOT/data/legacy-screen-expansion-v1/report.json"
legacy_sha=$(sha256sum "$legacy_report" | cut -d' ' -f1)
PYTHONPATH="$ROOT/overlay" "$PY" "$ROOT/source/old-source-overlap-expansion-v1/scan.py" \
  --manifest "$ROOT/source/old-source-overlap-expansion-v1/old-source-manifest.json" \
  --expected-manifest-sha256 8291e6c2a9af017904a3cc17b838cff231ff1a6da4cf7b9fe810b2dd6995a6e5 \
  --index-report "$legacy_report" --expected-index-report-sha256 "$legacy_sha" \
  --output "$ROOT/data/old-source-overlap-expansion-v1" --workers 16 \
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
  --raw-report "$raw_report" --expected-raw-report-sha256 "$raw_sha" \
  --exclusions "$exclusions" --expected-exclusions-sha256 "$exclusions_sha" \
  --audit-source "$ROOT/source/raw-audit-expansion-v1" \
  --output "$ROOT/data/quality-family-expansion-v1" --workers 16 \
  >"$LOG_ROOT/quality.log" 2>&1

features="$ROOT/data/quality-family-expansion-v1/report.json"
features_sha=$(sha256sum "$features" | cut -d' ' -f1)
PYTHONPATH="$ROOT/overlay-quality-family-v1" \
"$PY" "$ROOT/source/family-split-expansion-v1/scan.py" \
  --features-report "$features" --expected-features-sha256 "$features_sha" \
  --raw-report "$raw_report" --exclusions "$exclusions" \
  --design "$ROOT/source/fast-start-nosynthetic-v1/experiment-design.json" \
  --expected-design-sha256 3d58055a1d2398b90c963c9ed4be7e99e1d1644281b95599c4ad5339d805d105 \
  --output "$ROOT/data/family-split-expansion-v1" --workers 4 \
  >"$LOG_ROOT/family.log" 2>&1

family="$ROOT/data/family-split-expansion-v1/report.json"
family_sha=$(sha256sum "$family" | cut -d' ' -f1)
"$PY" "$ROOT/source/pack-selected-expansion-v1/pack.py" \
  --family-report "$family" --expected-family-sha256 "$family_sha" \
  --tokenizer "$ROOT/formal/hf-v5/tokenizer.json" \
  --expected-tokenizer-sha256 30d71356c5ba154006df5bbb4a0583fc434525ceeb2f27a7d8a237ce5db26dc6 \
  --output "$ROOT/data/selected-packs-expansion-v1" --workers 4 \
  >"$LOG_ROOT/pack.log" 2>&1

"$PY" - <<'PY'
import datetime, hashlib, json
from pathlib import Path
root = Path('/ssd/scxi253/pretrain500m-20260912-v1')
artifacts = [root/'data/diverse-audit-expansion-v1/report.json',
 root/'data/contamination-expansion-v1/report.json', root/'data/legacy-screen-expansion-v1/report.json',
 root/'data/old-source-overlap-expansion-v1/report.json', root/'receipts/exclusion-union-expansion-v1.json',
 root/'data/quality-family-expansion-v1/report.json', root/'data/family-split-expansion-v1/report.json',
 root/'data/selected-packs-expansion-v1/report.json']
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'status':'EXPANSION_FILTER_AND_PACK_COMPLETE_NOT_TRAINING_ADMITTED',
 'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'artifacts':{str(p):digest(p) for p in artifacts},'training_admitted':False,'training_launched':False}
(root/'receipts/pipeline-expansion-v1-complete.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
PY
