#!/usr/bin/env bash
set -euo pipefail

ROOT=/ssd/scxi253/pretrain500m-20260912-v1
PY=/ssd/scxi253/pretraining2/runtime/protrek-venv/bin/python
LOG="$ROOT/logs/pipeline-dclm-topup-v1"
mkdir -p "$LOG"

for pid_file in raw-audit-dclm-topup-v1-r2.pid old-source-overlap-dclm-topup-v1.pid; do
  pid=$(cat "$ROOT/receipts/$pid_file")
  while kill -0 "$pid" 2>/dev/null; do sleep 5; done
done

"$PY" - <<'PY'
import json
from pathlib import Path
root=Path('/ssd/scxi253/pretrain500m-20260912-v1')
expected={
 'data/diverse-audit-dclm-topup-v1/report.json':'COMBINED_RAW_INTAKE_MEASURED_NOT_ADMITTED',
 'data/contamination-dclm-topup-v1/report.json':'SUPPLEMENTAL_EXACT_SPAN_AUDIT_COMPLETE_ADMISSION_PENDING',
 'data/legacy-screen-dclm-topup-v1/report.json':'LEGACY_HASH_SCAN_COMPLETE_NOT_ADMITTED',
 'data/old-source-overlap-dclm-topup-v1/report.json':'OLD_SOURCE_NORMALIZED_OVERLAP_COMPLETE_NOT_ADMITTED'}
for rel,status in expected.items():assert json.loads((root/rel).read_text())['status']==status,(rel,status)
PY

exclusions="$ROOT/receipts/exclusion-union-dclm-topup-v1.json"
if [ -e "$exclusions" ]; then
  "$PY" - "$exclusions" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))['status']=='POLICY_EXCLUSIONS_BOUND_NOT_APPLIED_TO_PACK'
PY
else
  "$PY" "$ROOT/source/exclusion-union-expansion-v1/build.py" \
    --prior "$ROOT/receipts/exclusion-union-expansion-v1.json" \
    --supplemental-report "$ROOT/data/contamination-dclm-topup-v1/report.json" \
    --legacy-report "$ROOT/data/legacy-screen-dclm-topup-v1/report.json" \
    --old-source-report "$ROOT/data/old-source-overlap-dclm-topup-v1/report.json" \
    --output "$exclusions" >"$LOG/exclusion-union.log" 2>&1
fi

raw="$ROOT/data/diverse-audit-dclm-topup-v1/report.json"
raw_sha=$(sha256sum "$raw" | cut -d' ' -f1)
exclusions_sha=$(sha256sum "$exclusions" | cut -d' ' -f1)
prior_quality="$ROOT/data/quality-family-expansion-v1"
prior_quality_sha=$(sha256sum "$prior_quality/report.json" | cut -d' ' -f1)
PYTHONPATH="$ROOT/overlay-quality-family-v1" \
"$PY" "$ROOT/source/quality-family-expansion-v1/scan.py" \
  --raw-report "$raw" --expected-raw-report-sha256 "$raw_sha" \
  --exclusions "$exclusions" --expected-exclusions-sha256 "$exclusions_sha" \
  --audit-source "$ROOT/source/raw-audit-expansion-v1" \
  --output "$ROOT/data/quality-family-dclm-topup-v1" --workers 1 \
  --reuse-output "$prior_quality" --expected-reuse-report-sha256 "$prior_quality_sha" \
  >"$LOG/quality.log" 2>&1

quality="$ROOT/data/quality-family-dclm-topup-v1/report.json"
quality_sha=$(sha256sum "$quality" | cut -d' ' -f1)
PYTHONPATH="$ROOT/overlay-quality-family-v1" \
"$PY" "$ROOT/source/family-split-expansion-v1/scan.py" \
  --features-report "$quality" --expected-features-sha256 "$quality_sha" \
  --raw-report "$raw" --exclusions "$exclusions" \
  --design "$ROOT/source/fast-start-nosynthetic-v1/experiment-design.json" \
  --expected-design-sha256 3d58055a1d2398b90c963c9ed4be7e99e1d1644281b95599c4ad5339d805d105 \
  --output "$ROOT/data/family-split-dclm-topup-v1" --workers 4 \
  >"$LOG/family.log" 2>&1

family="$ROOT/data/family-split-dclm-topup-v1/report.json"
family_sha=$(sha256sum "$family" | cut -d' ' -f1)
prior_pack="$ROOT/data/selected-packs-expansion-v1/report.json"
prior_pack_sha=$(sha256sum "$prior_pack" | cut -d' ' -f1)
"$PY" "$ROOT/source/pack-selected-incremental-v1/pack.py" \
  --family-report "$family" --expected-family-sha256 "$family_sha" \
  --tokenizer "$ROOT/formal/hf-v5/tokenizer.json" \
  --expected-tokenizer-sha256 30d71356c5ba154006df5bbb4a0583fc434525ceeb2f27a7d8a237ce5db26dc6 \
  --reuse-report "$prior_pack" --expected-reuse-report-sha256 "$prior_pack_sha" \
  --output "$ROOT/data/selected-packs-dclm-topup-v1" --workers 4 \
  >"$LOG/pack.log" 2>&1

"$PY" - <<'PY'
import datetime,hashlib,json
from pathlib import Path
root=Path('/ssd/scxi253/pretrain500m-20260912-v1')
artifacts=[root/'data/diverse-audit-dclm-topup-v1/report.json',root/'data/contamination-dclm-topup-v1/report.json',
 root/'data/legacy-screen-dclm-topup-v1/report.json',root/'data/old-source-overlap-dclm-topup-v1/report.json',
 root/'receipts/exclusion-union-dclm-topup-v1.json',root/'data/quality-family-dclm-topup-v1/report.json',
 root/'data/family-split-dclm-topup-v1/report.json',root/'data/selected-packs-dclm-topup-v1/report.json']
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'status':'DCLM_TOPUP_FILTER_AND_PACK_COMPLETE_NOT_TRAINING_ADMITTED',
 'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'artifacts':{str(p):digest(p) for p in artifacts},'training_admitted':False,'training_launched':False}
(root/'receipts/pipeline-dclm-topup-v1-complete.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
PY
