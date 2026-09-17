# MoE7B 150B data-build execution record

## Frozen build

- Build contract: `data/moe7b_150b_build_v1.json`
- Mixture contract: `data/moe7b_150b_mixture_v1.json`
- Immutable source inventory: `data/source-lock-20260917-v2/`
- Exact tokenizer SHA-256: `30d71356c5ba154006df5bbb4a0583fc434525ceeb2f27a7d8a237ce5db26dc6`
- Public content-free contamination index SHA-256: `5d986c549213c8bcd74f30e475461f7b76042398415e1cb844c152b87a41314d`
- Required training prediction tokens: 150,000,000,000, allocated exactly across Stages A/B/C by the final stage manifest.

The producer verifies every frozen LFS object, applies source quality and language gates, redacts web email/valid-IPv4 patterns, rejects code with email/valid-IPv4/secret patterns, scans exact and 13-gram public-evaluation hashes, performs global exact document deduplication, performs MinHash-LSH lexical near-deduplication, uses the frozen 50,280-token tokenizer, and writes SHA-256-receipted little-endian uint16 blocks.

## ParaCloud execution

- Immutable deployed source: `/ssd/scxi253/cvcr-research/source-data-20260917-v1`
- Deployed source manifest: `/ssd/scxi253/cvcr-research/source-data-20260917-v1/SOURCE_SHA256SUMS`
- Packed/state root: `/ssd/scxi253/cvcr-research/data/moe7b-150b-v1/production-v1`
- Raw verified cache: `/data/run01/scxi253/moe7b-150b-v1/raw-production-v1`
- Login-node download feeder status: `/ssd/scxi253/cvcr-research/data/moe7b-150b-v1/source-feeder-status.json`
- Slurm build job: `1595837` (`moe7b-data-150b`, 1x RTX 4090 allocation, 6 CPUs, 30-day limit, requeue enabled)

Compute nodes have no outbound HTTPS connectivity. The implementation therefore uses a bounded login-node feeder that keeps three verified files ahead and an offline compute consumer. Files become visible to the consumer only after their byte length and frozen LFS SHA-256 pass and an atomic verification marker is written.

## Validation completed before launch

- Local full project test suite: 30 passed (the focused data-build subset was 12 passed).
- Remote environment: Python 3.12.13, NumPy, pyarrow 20.0.0, tokenizers, and zstandard import successfully.
- Full preflight: all 12 component ids, source locks, tokenizer identity, contamination index identity, and target totals passed.
- Real peS2O shard smoke: 64 documents, 378,613 token ids; privacy, contamination, MinHash, and tokenizer paths passed.
- Six-process plus batched-tokenizer smoke: approximately 485,078 token ids/s on the bounded sample. This is a smoke throughput, not a sustained whole-corpus claim.
- Live handoff evidence: the first 4,098,179,021-byte peS2O object was downloaded and hash-verified; job `1595837` began writing its first train shard.

## Claim boundary and remaining admission gates

This is a running candidate-data build, not an admitted 150B pack and not evidence that training has started. Final training remains blocked until every component reaches its independent token quota, all packed receipts verify, the exact stage manifest is generated, final license/attribution review passes, semantic/paraphrase deduplication is independently audited, public evaluation coverage includes the missing exact RULER variant, a clean-room contamination audit passes, language/domain/human content audits pass, and train/validation/test isolation is evidenced.

## Prepared v7 throughput correction

V7 composes the measured six-CPU ordered analysis pipeline from v6 with batched
exact-hash and MinHash-LSH queries.  It preserves the scalar 0.75 similarity
and 256-row hot-bucket rules while replacing up to five shared-filesystem
SQLite round trips per document with bounded set queries per input batch.  The
batch lookup path passed explicit exact, near, far, hot-bucket, and 120 seeded
differential cases.  A real-shard v6/v7 comparison is still required before the
active writer changes source.

## Prepared v8 receipt-commit correction

The first production receipt exposed a second shared-storage bottleneck: about
460,000 accepted documents expand to about 2.3 million document and LSH index
inserts.  V1 issues these SQLite statements one at a time.  V8 preserves the
single FULL-synchronous transaction and uniqueness constraints but sends
document and band rows through bounded 4,096-record `executemany` batches.  Its
test crosses the batch boundary with 5,000 documents, verifies 20,000 LSH rows,
and verifies idempotent receipt reconciliation.  The active transaction remains
untouched until it commits or fails naturally.
