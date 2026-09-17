# Genuine 7B MoE programme

This directory records the reproducible source candidates for the 7.077B-total,
1.264B-active, 16-expert Top-2 model.  The current 8.376B-token FineWeb-Edu run
is a systems and stability pilot.  Its tokens are outside the formal budget and
it does not establish model quality.

## Live systems evidence at 2026-09-17 18:47 +08:00

- Job `1596079`: 8x RTX 5090, step 2590, 339,476,480 prediction tokens,
  11,300.6 token/s, 5.103% causal-useful MFU, finite loss/gradient, and 19.59 GB
  peak reserved memory per GPU.
- Job `1596075`: 16x RTX 4090 fallback, step 310, 40,632,320 tokens,
  1,283.2 token/s and 0.735% causal-useful MFU.  It remains a slow fallback,
  not a performance candidate.
- Job `1595837`: sole writer for the frozen 150B candidate pack.  The first
  peS2O receipt exists; its current v1 process is bulk-populating the durable
  SQLite exact/near-dedup state and has not yet advanced `progress.json`.

The latest durable step-1 checkpoints were preserved with hard-link archives,
so the archives add no second physical copy of roughly 56.6 GB each:

| Job | Archive receipt SHA-256 | Checkpoint manifest SHA-256 |
| --- | --- | --- |
| `1596079` | `c13b69ff2c93773892afac7f14486e05a873812ba50c6405bd8b5f9087baebef` | `034f27fb815b06ab9e7f8337b68d452c85b6633a8863a81fb726794cf4429b0e` |
| `1596075` | `789d579e36f6bc6c82311cef52ba59c0aee748a7bf278827d84171e69643ac63` | `a1591b82487e536361a26293f395c2065f6a6a9f1405701efb9a1cfa5bc89be7` |

The running process saves every 5,000 optimizer steps.  Forcing an in-memory
save would make its wrapper exit and requeue, risking the active allocation, so
the monitor will archive and hash the step-5000 checkpoint after it commits
online.

## Throughput candidates

`source-opt-v51` is deployed read-only at
`/ssd/scxi253/cvcr-research/source-opt-20260917-v51`.  Its
`SOURCE_SHA256SUMS` SHA-256 is
`22687267930c7af09406c6e4994eb1f8368e295b8d6cb67be52072417e5e1148`.
It adds bounded tests for retaining unsharded BF16 parameters between
accumulation microbatches, retaining the FSDP root group through backward, and
BF16 reduction as a separately gated numerical change.  Remote Torch 2.11
validation passed 31 tests with one CUDA-only test skipped.

Queued 8xRTX5090 screens are:

- `1598263/1598264/1598265`: microbatch/accumulation `2/4`, `4/2`, `8/1`.
- `1598276`: keep-unsharded `1/8`, FP32 reduction.
- `1598277`: keep-unsharded `2/4`, FP32 reduction.
- `1598278`: keep-unsharded `1/8`, BF16 reduction.

Every screen keeps 131,072 prediction tokens per optimizer step.  Promotion
requires finite metrics, at least 2 GiB per-GPU headroom, at least 10% median
throughput improvement over 11,260 token/s, matched all-tensor update parity,
and sustained confirmation.  No candidate changes the active checkpoint
fingerprint or silently replaces a running job.

## Formal 150B data and launch

The frozen target is exactly 150,000,000,000 prediction tokens, with at least
142.5B unique tokens and no more than 5% replay.  The aggregate allocation is
61B FineWeb-Edu, 26B DCLM, 15B multilingual, 25B permissive code, 15.5B math,
and 7.5B science/reference, arranged in 100B/40B/10B stages.

`source-data-v8` composes the measured ordered analysis/tokenization pipeline,
batched exact and LSH queries, and 4,096-record SQLite `executemany` receipt
commits.  It is deployed read-only at
`/ssd/scxi253/cvcr-research/source-data-20260917-v8`; its manifest SHA-256 is
`907b98423936667387d39f8d848573aef1f843688ee39afe328a6e98a9f95493`.
Remote validation passed 12 tests plus exact/near/far/hot-bucket and 120 seeded
differential cases.  Job `1598301` is the real-shard v8 benchmark.

Formal training starts from fresh initialization through
`source-opt-v51/scripts/submit_formal_150b.sh`.  The script refuses submission
until the exact stage manifest and `TRAINING_ADMITTED` receipt agree on all
150B tokens and every license, provenance, privacy, deduplication,
contamination, quality, language/domain, and split-isolation gate is closed.
