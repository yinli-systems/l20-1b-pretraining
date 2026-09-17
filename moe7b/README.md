# Genuine 7B MoE programme

This directory contains the evidence-bounded programme for a 7.077B-total,
1.264B-active language MoE. The current model has 16 experts with Top-2
routing. Its 8.376B-token FineWeb-Edu jobs are systems and stability pilots;
their tokens remain outside the formal quality budget.

## Live state at 2026-09-17 20:16 +08:00

- Job `1596079` is running on 8x RTX 5090. At step 3,050 it had processed
  399,769,600 prediction tokens at 11,231.5 token/s, 5.072% causal-useful MFU,
  finite loss and gradient, and 19.59 GB peak reserved memory per GPU. This is
  the useful stability run, but it is far below the 50% production MFU gate.
- Job `1596075` is running on 16x RTX 4090 as a slow fallback. At step 360 it
  had processed 47,185,920 tokens at 1,283.6 token/s and 0.735% MFU. It is not
  a performance candidate.
- Job `1595837` is the sole 150B data writer. Its SQLite WAL reached
  137,570,952 bytes at 20:16, but `production-v1/progress.json` still reports
  zero available tokens and `RUNNING_NOT_ADMITTED`. No formal training can
  consume this data yet.
- Job `1598301`, the real-shard data-pipeline v8 benchmark, is pending with a
  current scheduler estimate of 21:47. The estimate can move.
- Job `1598354`, the immutable v54 8-way expert-parallel screen, is pending.
  Its current scheduler estimate is 2026-09-25 10:59.
- Job `1598474`, the immutable v55 4-way expert-parallel screen, is pending.
  Its current scheduler estimate is 2026-09-18 00:39.
- The public `gpu_4090`, `gpu_5090`, `hp_4090`, and `hp_5090` partitions
  currently report zero available GPUs. The project nevertheless retains the
  25 GPUs held by its two pilots and data writer.

The latest durable step-1 checkpoints were preserved with hard-link archives,
so the archives add no second physical copy of roughly 56.6 GB each:

| Job | Archive receipt SHA-256 | Checkpoint manifest SHA-256 |
| --- | --- | --- |
| `1596079` | `c13b69ff2c93773892afac7f14486e05a873812ba50c6405bd8b5f9087baebef` | `034f27fb815b06ab9e7f8337b68d452c85b6633a8863a81fb726794cf4429b0e` |
| `1596075` | `789d579e36f6bc6c82311cef52ba59c0aee748a7bf278827d84171e69643ac63` | `a1591b82487e536361a26293f395c2065f6a6a9f1405701efb9a1cfa5bc89be7` |

The pilots save every 5,000 optimizer steps. Forcing an early in-memory save
would make the wrapper exit and requeue, releasing the active allocation. The
monitor therefore preserves and hashes normal online checkpoints after they
commit.

## Quality programme

The research-backed candidate budget is 5T base pretraining tokens plus a
separately admitted 100B high-quality anneal. Checkpoints at 150B, 300B, 600B,
1T, 2.6T, 5T, and 5.1T are predeclared stop/revise gates on one schedule. The
current 150B corpus build is the first data and recipe qualification slice,
not the final token budget.

The machine-readable programme is
[`plans/high_quality_5p1t_program_v1.json`](plans/high_quality_5p1t_program_v1.json).
The evidence and primary-source rationale are in
[`reports/high_quality_training_research_20260917.md`](reports/high_quality_training_research_20260917.md).
The plan validator checks token arithmetic, mixture sums, checkpoint ordering,
and promotion thresholds.

Before freezing a production mixture, three 529M proxy arms compare a
web-heavy mix, the current 150B mix, and a capability-heavy mix for exactly
536,870,912 prediction tokens each, followed by two-seed confirmation. The F2
data continuation is retired because its seven-task mean regressed from
48.53115% to 48.45525%. Its high-MFU engineering lessons remain applicable:
large microbatches, low accumulation, constant tokens per update, deterministic
resume, and rolling performance gates.

The formal quality threshold is a seven-domain macro above 50%. Exceeding a
named model additionally requires the same weights scope, tokenizer, prompts,
support setting, sample rows, and scoring code, with a positive paired-bootstrap
lower bound against every named comparator. The budget and thresholds are a
research contract, not a claim that the data is admitted or that the model
already exceeds another model.

## Throughput qualification

`source-opt-v55` is deployed read-only at
`/ssd/scxi253/cvcr-research/source-opt-20260917-v55`. Its
`SOURCE_SHA256SUMS` SHA-256 is
`e4ade692d91825418d83415cf4483a4beeddb19bb7fabd4dc2f6c73d772d50cf`.
Local validation passed 32 tests with two environment-specific skips. Remote
Torch 2.11 validation passed 33 tests with one CUDA-only skip.

The v54/v55 expert-parallel screens preserve exactly 131,072 prediction tokens
per optimizer step while testing 8-way and 4-way expert placement. Each fails
closed unless metrics are finite, per-GPU headroom is at least 2 GiB, and the
post-warmup median causal-useful MFU is at least 50%. A pass permits the next
numerical qualification stage only: all-tensor update parity, FP32 optimizer
semantics, checkpoint/reload parity, and a sustained 500-step confirmation.

The next bounded performance candidates are grouped expert GEMM, fused
permutation/router kernels, all-to-all overlap, and a distributed optimizer.
Data-parallel replicas are added only after expert-parallel correctness and
deterministic sample order are frozen. Architecture proxy tests must compare
16/2, 32/4, and 64/8 expert granularity at matched total and active capacity;
an architecture change requires a fresh formal initialization.

## Data admission

`source-data-v8` is deployed read-only at
`/ssd/scxi253/cvcr-research/source-data-20260917-v8`; its manifest SHA-256 is
`907b98423936667387d39f8d848573aef1f843688ee39afe328a6e98a9f95493`.
Remote validation passed 12 tests plus exact, near, far, hot-bucket, and 120
seeded differential cases.

Every formal document must bind source revision, byte hash, license and
provenance, quality and privacy decisions, and split. Admission also requires
global URL, exact, near-document, and repeated-line deduplication; benchmark
contamination checks; sealed-test isolation; tokenizer and packed-shard
identities; and exact next-token counts. Padding, validation, failed,
recomputed, and pilot tokens do not count.

The current writer remains fail-closed until every receipt advances to
`TRAINING_ADMITTED`. The existing 150B launcher will therefore refuse to start
training today. Expansion toward 5.1T proceeds only after the proxy mixture,
data identities, systems path, and first 150B gate all pass.

The raw-corpus expansion now has a second independent entry point for the
official Nemotron-CC release. Its frozen inventory contains 31,279 compressed
JSONL objects and reports 6.3T tokens in 10.4 TiB: 4.4T globally deduplicated
real tokens plus 1.9T synthetic tokens. Acquisition prioritizes high and
medium-high partitions, validates the official path-index identity, tests each
zstd stream, hashes every object, and stops before the 20 TiB free-space floor.
A one-object smoke job must pass before the full resumable job can run.

These are raw source tokens and are not automatically part of the 5.1T budget.
Admission still requires document identities, provenance and license review,
PII/secrets and safety handling, global cross-source deduplication, benchmark
contamination checks, exact frozen-tokenizer counts, split isolation, quality
stratification, and deterministic human audits. The acquisition contract and
implementation are
[`nemotron_cc_acquisition_protocol_v1.json`](nemotron_cc_acquisition_protocol_v1.json)
and [`acquire_nemotron_cc.py`](acquire_nemotron_cc.py).

The completed v8 real-shard benchmark (`1598301`) processed approximately
64.15M token IDs in 242 seconds, about 265k token IDs/s for one pipeline. This
proves the current serial builder is far too slow for a 5.1T programme. The
production transformation stage must therefore shard download, filtering,
deduplication and tokenization before it can issue a training-admission
receipt.

Nemotron-CC jobs `1598652`/`1598653` are the one-object smoke and dependent
full-inventory pair. They use immutable source
`/ssd/scxi253/data-scale-source-20260917-v1`; its source-manifest SHA-256 is
`de4f5b0412bf4354f10f6b75809fc2bb5f564f28726bcc2fc72f4924ccb114d3`.
The smoke was scheduler-accepted and pending priority at submission. The full
job cannot start until smoke exits successfully.

## Compute boundary

With the project's strict useful-FLOP numerator and RTX 5090 denominator, a
5.1T run takes an idealized 533 days on 8 cards at 50% MFU, 266 days on 16
cards, or 66.6 days on 64 cards. At 77% MFU the corresponding estimates are
346, 173, and 43.3 days. Queueing, evaluation, checkpointing, data I/O, and
failures add time, so GPU scale and measured checkpoint gates are essential.
