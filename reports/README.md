# Evidence bundle

This directory contains small, immutable receipts and derived metrics from the
single-L20 production run. Training and the registered external evaluation are
complete; the files report measured evidence without a cross-model superiority claim.

- `receipts/` records the exact software/hardware environment, data token
  counts and hashes, gate outcome, selected micro-batch benchmark, and the
  authorized pre-run cleanup boundary.
- `progress/` records point-in-time live telemetry. A snapshot does not update
  automatically and must not be interpreted as the current state after its
  `captured_at` timestamp.
- `metrics/validation-loss.csv` contains every completed 500-step validation.
- `metrics/training-loss-100-step.csv` contains 100-step means and 10th/90th
  percentile bands derived from TensorBoard.
- `metrics/loss-forecast.csv` is the documented late-window power-law
  extrapolation, not a measured result or statistical confidence interval.
- `metrics/final-benchmarks.json` is the compact final metric summary. The large
  raw lm-eval JSON files remain on the GPU host and are bound by hashes in
  `receipts/final-evaluation-receipt.json`.
- `research/tinyllama-comparison-20260911.md` documents the added BoolQ evaluation
  and revision-pinned TinyLlama 3T/v1.1 comparison, including protocol and claim
  boundaries. All seven comparison jobs are complete.
- `metrics/tinyllama-comparison-20260911.json` contains the full compact metrics,
  paired bootstrap intervals, alignment fingerprints, answer-bias diagnostics,
  and post-evaluation model/tokenizer hashes. Its matching receipt is
  `receipts/tinyllama-comparison-20260911.json`; large per-sample JSON remains
  on the GPU host and is bound by SHA-256.
- `research/efficiency-baseline-execution-20260912.md` records the new
  same-protocol baseline work: 36 declared checkpoint evaluations, existing
  reference reuse, lower-compute controls, adapter validation, and access
  blockers.
- `metrics/efficiency-all-results-final-20260912.md` and its companion JSON are
  the completed 36/36 checkpoint comparison, including the seven-task and
  six-task sensitivity results, paired bootstrap intervals, hashes, and claim
  boundaries. The frozen subset is not a world ranking.
- `plots/efficiency-frontier-clean-20260912.{png,pdf,svg}` is the publication
  figure generated from the completed result bundle.
- `receipts/efficiency-evaluation-plan-20260912-v3.json` freezes that work list,
  immutable model revisions, software/artifact hashes and the protocol.
  Earlier inventory/plan versions are retained as provenance.
- `receipts/efficiency-training-pause-20260912.json` verifies B's user-requested
  resumable step-61 pause, allowing evaluation to take priority without losing
  completed optimizer updates. No automatic training resume is scheduled.
- `metrics/datadecide-upstream-scan-20260912.json` records the separate
  25-recipe upstream OLMES selection audit, not our matched-harness results.
- `research/2b-continuation-plan.md` reviews primary research and proposes a
  strictly budgeted continuation experiment. Subsequent recipe revisions and
  actual execution are recorded in the execution protocol.
- `research/1b-posttraining-deep-research-20260912.md` validates the proposed
  distillation/SFT/preference/RLVR/re-distillation route against primary sources,
  corrects its single-L20 compute assumptions, and defines the Stage-0 access
  boundary. The matching machine-readable contract is
  `../posttraining_protocol.json`; no post-training run is claimed started.
- `receipts/posttraining-protocol-validation-20260912.json` binds the first
  protocol implementation and local test result. It also records that GitHub
  Actions did not execute any step because of an account billing/spending-limit
  gate; this is infrastructure failure, not a green or failed code test.
- `receipts/posttraining-space-cleanup-20260912.json` records the exact,
  hash-bound removal of one obsolete 20-step smoke checkpoint to make room for
  continuation B's atomic resume save. Production, gate, continuation,
  evaluation, and published weights were explicitly preserved. The constrained
  cleanup implementation is `../cleanup_obsolete_smoke_checkpoint.py`.
- `research/continuation-execution-protocol.md` records the follow-up research,
  isolated implementation, budget ledger, data/quality gates, and current
  execution boundary. Pilot A completed all 190 steps; this is not an automatic
  main-run promotion.
- `research/continuation-pilot-evaluation-20260911.md` records A completion,
  frozen-checkpoint export/parity, same-protocol evaluation, and B preparation.
- `metrics/continuation-A-core-20260911.json` is the complete core comparison;
  `receipts/continuation-evaluation-A-20260911.json` is its hash-matched receipt
  snapshot while extended evaluation was still running, not a live endpoint.
- `receipts/continuation-export-A-20260911.json` binds exported weights,
  tokenizer/configuration files and the forward-only numerical check.
- `metrics/continuation-A-all-20260911.json` and
  `receipts/continuation-evaluation-A-complete-20260911.json` contain the completed
  full comparison and its matching final receipt. A is not promoted to the main run.
- `receipts/continuation-quality-review-B-20260911.json` binds the final B data,
  expanded excerpt audit and limitations; it admits only the 190-step B pilot.
- `receipts/continuation-launch-B-20260911.json`,
  `receipts/continuation-resource-preflight-B-20260911.json`,
  `receipts/continuation-input-verification-B-20260911.json`, and
  `receipts/continuation-pilot-B-start-20260911.json` document B admission,
  launch/resource checks and the first completed optimizer update.
- `receipts/continuation-preflight-20260911.json` binds the original native/HF
  weights and tokenizer and records fixed-probe validation and numerical checks.
- `receipts/continuation-quality-review-A-20260911.json` binds reviewed data,
  source/program hashes, sample hashes and the limits of the assistant's audit.
- `receipts/continuation-launch-A-20260911.json` and
  `receipts/continuation-resource-preflight-A-20260911.json` record the frozen
  training configuration and validated child-process file-limit adjustment.
- `receipts/continuation-pilot-A-start-20260911.json` is a dated early-training
  snapshot, not a live status endpoint or a new held-out validation result.
- `figures/loss-curve-step-14094.png` is the latest curve snapshot; earlier
  figures are retained as immutable historical evidence.

Raw corpora, caches, deduplication databases, logs, TensorBoard event files,
and checkpoints are intentionally excluded. They are large, mutable, or may
contain source text. Model weights will use a model-artifact release after the
training and evaluation gates complete, rather than normal Git history.

`environment-receipt.json` was captured before the final data-pipeline and
launcher fixes, so its package/hardware fields remain useful while its embedded
`project_sha256` map is historical. Use
`production-source-comparison.json` for the post-launch comparison: 18 of 19
production files match byte-for-byte, while `hf_config.py` differs only by one
trailing blank line and has the same normalized SHA-256.
