# Evidence bundle

This directory contains small, immutable receipts and derived metrics from the
single-L20 production run. The files are evidence snapshots, not claims that
the unfinished model has passed downstream benchmarks.

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
- `figures/loss-curve-step-10576.png` visualizes the corresponding snapshot.

Raw corpora, caches, deduplication databases, logs, TensorBoard event files,
and checkpoints are intentionally excluded. They are large, mutable, or may
contain source text. Model weights will use a model-artifact release after the
training and evaluation gates complete, rather than normal Git history.
