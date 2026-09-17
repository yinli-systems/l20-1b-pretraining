#!/bin/bash
set -euo pipefail

SOURCE="${SOURCE:-/ssd/scxi253/cvcr-research/source-opt-20260917-v51}"
SCATTERMOE_SOURCE="${SCATTERMOE_SOURCE:-/ssd/scxi253/cvcr-research/vendor/scattermoe-47b5e150}"
STAGE_MANIFEST="${STAGE_MANIFEST:-/ssd/scxi253/cvcr-research/data/moe7b-150b-v1/production-v1/stage-mixture-manifest-v1.json}"
DATA_ADMISSION="${DATA_ADMISSION:-/ssd/scxi253/cvcr-research/data/moe7b-150b-v1/production-v1/training-admission-v1.json}"
RUN_NAME="${RUN_NAME:-moe7b-top2-scattermoe-fsdp2-8x5090-formal150b-s20260917-v1}"
RUN_ROOT="${RUN_ROOT:-/ssd/scxi253/cvcr-research/runs}"
MICROBATCH="${MICROBATCH:-1}"
ACCUMULATION="${ACCUMULATION:-8}"
KEEP_UNSHARDED="${KEEP_UNSHARDED:-0}"
KEEP_ROOT_UNSHARDED="${KEEP_ROOT_UNSHARDED:-0}"
REDUCE_DTYPE="${REDUCE_DTYPE:-float32}"
TARGET_TOKENS=150000000000
SCRIPT="$SOURCE/slurm/moe_7b_fsdp2_scattermoe_8x5090.sbatch"

test -f "$SOURCE/SOURCE_SHA256SUMS"
(cd "$SOURCE" && sha256sum --check --quiet SOURCE_SHA256SUMS)
test -f "$STAGE_MANIFEST" -a -f "$DATA_ADMISSION"
test ! -e "$RUN_ROOT/$RUN_NAME"
"${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}" - \
  "$STAGE_MANIFEST" "$DATA_ADMISSION" "$TARGET_TOKENS" <<'PY'
import hashlib
import json
import pathlib
import sys

stage_path = pathlib.Path(sys.argv[1])
admission_path = pathlib.Path(sys.argv[2])
target = int(sys.argv[3])
stage_bytes = stage_path.read_bytes()
stage = json.loads(stage_bytes)
admission = json.loads(admission_path.read_text())
assert admission["status"] == "TRAINING_ADMITTED"
assert admission["training_admitted"] is True
assert admission["stage_manifest_sha256"] == hashlib.sha256(stage_bytes).hexdigest()
assert admission.get("open_gates") in ([], None)
assert int(stage["exact_prediction_tokens"]) == target
assert int(admission["exact_prediction_tokens"]) == target
PY

sbatch --parsable \
  --job-name=moe7b-formal150b \
  --export=ALL,SOURCE="$SOURCE",SCATTERMOE_SOURCE="$SCATTERMOE_SOURCE",STAGE_MANIFEST="$STAGE_MANIFEST",DATA_ADMISSION="$DATA_ADMISSION",TARGET_TOKENS="$TARGET_TOKENS",RUN_NAME="$RUN_NAME",RUN_ROOT="$RUN_ROOT",RUN_ROLE=formal_pretraining,MICROBATCH="$MICROBATCH",ACCUMULATION="$ACCUMULATION",KEEP_UNSHARDED="$KEEP_UNSHARDED",KEEP_ROOT_UNSHARDED="$KEEP_ROOT_UNSHARDED",REDUCE_DTYPE="$REDUCE_DTYPE" \
  "$SCRIPT"
