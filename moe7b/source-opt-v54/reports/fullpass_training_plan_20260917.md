# Top-2 fast-path full-pass training plan — 2026-09-17

## Scope and claim boundary

This run trains the validated 8-layer proxy with ScatterMoE, Liger fused
linear cross-entropy, microbatch 32 per rank, accumulation 2, and four RTX
4090 GPUs.  It is not the 7.002B model and does not reverse the failed CVCR or
backend non-inferiority decisions.  The selected path is an explicit user
choice to maximize measured proxy throughput/MFU.

The existing 7B configuration cannot be launched by this DDP runtime: FP32
parameter storage alone is about 28 GB per rank before gradients and AdamW
states, and its 32,000-token vocabulary conflicts with the current 50,280
token data lineage.  Configuration validation now fails closed on the latter.

## Frozen run contract

- Model config: `configs/model/oracle_top2_scattermoe_liger.json`
- Source snapshot: `/ssd/scxi253/cvcr-research/source-opt-20260917-v19`
- ScatterMoE commit: `47b5e1502e5a10e82c8e5945d761b877849871e7`
- Liger wheel SHA-256:
  `84c0a7bc9bf4d4cf8ea5ba89ff84d28686afc94215b220851d9f57dc87852741`
- Frozen pack: `fineweb-edu-formal-v1`, status `FROZEN_VERIFIED_PACK`
- Available unique prediction tokens: 8,376,643,584
- Executed steps/tokens: 15,977 / 8,376,549,376
- Checkpoint and validation cadence: every 1,000 steps plus final
- Output directory:
  `/ssd/scxi253/cvcr-research/runs/proxy-top2-scattermoe-liger-mb32-fullpass-s20260917-v1`

The batch script requests eight hours and permits Slurm requeue.  A restart is
allowed only when both `resume.pt` and `resume.sha256` exist and match; a
non-empty run without a verified checkpoint fails closed.  Periodic
checkpoints bound ordinary interruption loss to at most 1,000 optimizer steps.

## Launch evidence

- Slurm job: `1595686`
- Allocation: four RTX 4090 GPUs on `wqd10naf05g5`
- Initial live window: steps 3--17, approximately 517k--525k tok/s and
  43.3%--43.9% causal-useful MFU
- First observed token count: 8,912,896 prediction tokens at step 17
- Monitoring heartbeat: `monitor-top2-full-pass-training`, every 15 minutes;
  quiet while healthy, with checksum-gated resubmission restricted to
  preemption, node failure, timeout, or `CHECKPOINTED_STOP`

Scheduler submission, allocation, live steps, checkpoint availability, and
completion remain separate states.  This receipt establishes only allocation
and live finite training; completion will require the final status, checkpoint
receipt, validation, and run-directory `SHA256SUMS`.
