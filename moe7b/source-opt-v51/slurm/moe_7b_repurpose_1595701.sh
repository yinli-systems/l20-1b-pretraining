#!/bin/bash
# Preserve the already scheduled 4x5090 backfill slot.  Slurm has already
# consumed the directives from the original submission; this wrapper only
# replaces its command and hands execution to the immutable v20 launcher.
set -euo pipefail
export SOURCE=/ssd/scxi253/cvcr-research/source-opt-20260917-v20
export SCATTERMOE_SOURCE=/ssd/scxi253/cvcr-research/vendor/scattermoe-47b5e150
# Two global optimizer steps (2 * 131,072 prediction tokens) are enough to
# exercise construction, forward/backward, full checkpoint, restore, and a
# post-restore update inside the reserved 35-minute qualification allocation.
export RUN_NAME=moe7b-top2-scattermoe-fsdp2-resume-gate-s20260917-v1
export TARGET_TOKENS=262144
exec bash "$SOURCE/slurm/moe_7b_fsdp2_scattermoe_4x5090.sbatch"
