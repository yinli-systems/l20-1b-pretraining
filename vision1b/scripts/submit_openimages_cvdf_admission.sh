#!/bin/bash
set -euo pipefail

SOURCE=${SOURCE:?set SOURCE to immutable admission source}
DATA=${DATA:-/ssd/scxi253/vision1b-research/data/openimages-cvdf-train-1p7m-v1}

(cd "$SOURCE" && sha256sum --check --quiet SOURCE_SHA256SUMS)
"${PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}" - "$DATA/acquisition-receipt.json" <<'PY'
import json
import sys

receipt = json.load(open(sys.argv[1]))
if receipt.get("status") != "DOWNLOADED_NOT_ADMITTED":
    raise SystemExit("full downloaded-not-admitted receipt required")
if receipt.get("requested_archives") != 16 or len(receipt.get("archives", [])) != 16:
    raise SystemExit("all 16 frozen archives are required")
PY

prep=$(SOURCE="$SOURCE" DATA="$DATA" sbatch --parsable --export=ALL,SOURCE="$SOURCE",DATA="$DATA" \
  "$SOURCE/slurm/openimages_cvdf_admission_prepare_1gpu.sbatch")
index=$(SOURCE="$SOURCE" DATA="$DATA" sbatch --parsable --dependency="afterok:$prep" --kill-on-invalid-dep=yes \
  --export=ALL,SOURCE="$SOURCE",DATA="$DATA" "$SOURCE/slurm/openimages_cvdf_admission_index_array_1gpu.sbatch")
reduce=$(SOURCE="$SOURCE" DATA="$DATA" sbatch --parsable --dependency="afterok:$index" --kill-on-invalid-dep=yes \
  --export=ALL,SOURCE="$SOURCE",DATA="$DATA" "$SOURCE/slurm/openimages_cvdf_admission_reduce_1gpu.sbatch")
audit=$(SOURCE="$SOURCE" DATA="$DATA" sbatch --parsable --dependency="afterok:$reduce" --kill-on-invalid-dep=yes \
  --export=ALL,SOURCE="$SOURCE",DATA="$DATA" "$SOURCE/slurm/openimages_cvdf_admission_audit_1gpu.sbatch")
pack=$(SOURCE="$SOURCE" DATA="$DATA" sbatch --parsable --dependency="afterok:$reduce" --kill-on-invalid-dep=yes \
  --export=ALL,SOURCE="$SOURCE",DATA="$DATA" "$SOURCE/slurm/openimages_cvdf_admission_pack_array_1gpu.sbatch")

printf 'PREP=%s\nINDEX=%s\nREDUCE=%s\nAUDIT=%s\nPACK=%s\n' "$prep" "$index" "$reduce" "$audit" "$pack"
printf 'FINALIZE_NOT_SUBMITTED: human and safety PASS receipts are still required.\n'
