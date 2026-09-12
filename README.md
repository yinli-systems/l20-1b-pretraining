# High-quality English 1.1B pretraining

[![CI](https://github.com/yinli-systems/l20-1b-pretraining/actions/workflows/ci.yml/badge.svg)](https://github.com/yinli-systems/l20-1b-pretraining/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)

> Part of [Pretraining Lab](https://github.com/yinli-systems/pretraining-lab),
> an evidence-first collection of from-scratch language models trained on one
> NVIDIA L20.

The released base checkpoint is available at
[`AliceYin/L20-1B-20B-Base`](https://huggingface.co/AliceYin/L20-1B-20B-Base).
The complete training code, frozen protocols, compact receipts, evaluation
artifacts, and plotting tools are maintained in this public repository.

This is a from-zero pretraining pipeline for one NVIDIA L20. It does not load a
pretrained model or tokenizer. The model is a 1,100,048,384-parameter
TinyLlama-class decoder (22 layers, width 2048, 32 attention heads, 4 query
groups, SwiGLU 5632, context 2048) and the tokenizer is a newly trained 32K
byte-level BPE.

Third-party pretrained weights are loaded only for separate baseline evaluations,
never to initialize this model's training.

## Results and evidence

![Training and validation loss through step 14094](reports/figures/loss-curve-step-14094.png)

The run completed all 19,148 optimizer steps and 19,999,703,040 prediction
tokens. Final held-out loss was 2.4247146 (perplexity 11.2990036). Immutable
environment, data, gate, benchmark, and progress receipts are indexed in
[`reports/`](reports/README.md). The final external benchmark summary is in
[`reports/metrics/final-benchmarks.json`](reports/metrics/final-benchmarks.json).
The revision-pinned, same-protocol TinyLlama comparison is documented in
[`reports/research/tinyllama-comparison-20260911.md`](reports/research/tinyllama-comparison-20260911.md).
The evidence review and strictly budgeted 2B-token continuation proposal are in
[`reports/research/2b-continuation-plan.md`](reports/research/2b-continuation-plan.md).
The evidence-checked, single-L20 post-training roadmap and its compute limits are in
[`reports/research/1b-posttraining-deep-research-20260912.md`](reports/research/1b-posttraining-deep-research-20260912.md).
Its fail-closed stage contract is [`posttraining_protocol.json`](posttraining_protocol.json);
it explicitly forbids automatic promotion and literature-scale RLVR without new
measured admission evidence.
Pilot A completed 190 steps (198,451,200 additional prediction tokens); fixed
validation PPL changed from 11.298834 to 11.297638, essentially flat. Its frozen
checkpoint completed the nine-task core evaluation: primary seven-task mean
51.01% versus the original 51.26%, with no confirmed improvement. Extended
same-protocol evaluation is also complete; A is not promoted to the main run.
Pilot B passed its bounded data-admission checks and completed its first real
optimizer update from the original 20B parent; this does not automatically approve
the main run. See the [execution protocol](reports/research/continuation-execution-protocol.md)
and [pilot evaluation record](reports/research/continuation-pilot-evaluation-20260911.md).
No post-continuation benchmark win is claimed.
The additional low-compute baseline evaluations and candidate-set Pareto analysis
are documented in the [baseline execution record](reports/research/efficiency-baseline-execution-20260912.md).
All 36 frozen checkpoint evaluations completed. The model's seven-task
same-protocol macro is 51.2601% (49.9411% excluding BoolQ). It significantly
beats the TinyLlama 1T checkpoint by +1.0182 percentage points, 95% paired
bootstrap CI [+0.1584, +1.8860], ties the 1.5T and 2T checkpoints, and loses to
the 2.5T checkpoint. The [complete result table and claim boundaries](reports/metrics/efficiency-all-results-final-20260912.md)
and [clean frontier figure](reports/plots/efficiency-frontier-clean-20260912.png)
are checked in. This frozen subset is not a global census or proof that the
model is universally first in token efficiency. Continuation B is a separate,
evidence-gated experiment and is not part of the released base checkpoint.
Measured validation points and the explicitly labeled pre-completion 20B-token
extrapolation are available as CSV files so the chart can be independently reproduced.
Checkpoint weights, source text, mutable logs, and raw TensorBoard events are
not stored in normal Git history.

## Data contract

The 20B-token production mixture is 42.5% FineWeb-Edu-Dedup score 4+, 42.5%
DCLM-Baseline, 3% FineMath-4+, and 12% permissively licensed Stack-Edu
Python/JavaScript/TypeScript/C++/Java at its release-validated educational
thresholds (score 3+; Java score 2+). This follows the short-horizon
foundation proportions supported by the SmolLM3 data ablations while keeping
the run English-only. Dataset revisions are pinned in `source_docs.py` and
`pipeline_config.json`; revision-bound Hub file lists are atomically cached so
resume does not depend on a fresh metadata API request. Cosmopedia remains only in the already-completed gate
and tokenizer sample; it is not part of the production 20B-token mixture.

Every accepted document passes English/content filters, global normalized-text
SHA-256 exact deduplication, and a 13-word benchmark decontamination check. The
PII status of DCLM is not documented, so DCLM email addresses and syntactically
valid IPv4 addresses are deterministically anonymized before deduplication and
tokenization. The validation split is a deterministic hash holdout. NumPy token shards are written
atomically with SHA-256 receipts and converted to LitData only after token-count
verification. Filtering and decontamination use six ordered worker processes,
tokenization uses exact-equivalent batched Rust encoding, Stack-Edu content uses
48 measured-concurrent SWH requests, and the next Parquet file is downloaded
while the current one is processed. Fully consumed raw
Parquet files are recorded in SQLite before
their cache copies are removed; the current partial file is retained for resume.

## Fail-closed execution

`build_data.py --stage gate` prepares four real-data samples. `run_gate.py` then
requires a successful 1,000-step run and checkpoint reload, finite validation
losses, at least 10K steady tokens/s, and at least 90% mean active GPU
utilization. `production_supervisor.py` refuses to create the full corpus or
start training unless those checks pass. It next requires exact agreement
between all source manifests, NumPy token counts, and packed LitData counts.

The full run uses BF16, compiled PyTorch SDPA, fused AdamW, micro-batch 6,
gradient accumulation 85, effective global batch 510 sequences, 1,000 optimizer
steps of warmup, and cosine decay from 4e-4 to 4e-5. It retains the two latest
complete step checkpoints and resumes automatically after interruption.

The hardware was one NVIDIA L20 with 46,068 MiB VRAM. A representative live
snapshot at step 10,595 measured 12,845 tokens/s, 93.948 TFLOP/s of model FLOPs,
100% GPU utilization, 45,265/46,068 MiB memory use, and 348.3 W. Using exactly
132 TFLOP/s as the BF16 peak denominator, `model_FLOP/s / 132e12`, this is
71.17% standard MFU. It is point-in-time telemetry, not a full-run average.

The packed reservoir contains at least 20.2B unique-source tokens (1% above the
nominal mixture; a resumed source may retain a larger verified reservoir).
Training is still capped at 19,999,703,040 effective prediction tokens,
exactly 19,148 complete optimizer steps. The reserve absorbs the 2049-input vs.
2048-target boundary and seeded weighted-sampling variance without cycling an
exhausted source.

## Operations

On the GPU host:

```bash
tmux list-sessions
tmux attach -t production
tmux attach -t production-watchdog
cat /home/hhai/pretrain/manifests/production-status.json
cat /home/hhai/pretrain/manifests/watchdog-status.json
cat /home/hhai/pretrain/manifests/training-gate-receipt.json
cat /home/hhai/pretrain/data/full-npy/math/progress.json
tail -f /home/hhai/pretrain/logs/production-supervisor.log
nvidia-smi dmon -s pucvmet
```

Important receipts live under `/home/hhai/pretrain/manifests`; logs are under
`/home/hhai/pretrain/logs`. The gate model is never used to initialize the full
run: `/home/hhai/pretrain/checkpoints/full` starts from random weights. The
final checkpoint is converted with the canonical `hf_config.py` architecture
and evaluated only after training has completed.

This pipeline maximizes evidence and quality within the fixed 20B-token budget.
The measured external results are reported without claiming superiority to
models trained on hundreds of billions or trillions of tokens.

## License

The code and repository-authored documentation are released under the
[MIT License](LICENSE). External datasets, baseline models, and generated model
artifacts remain subject to their respective licenses and terms.
