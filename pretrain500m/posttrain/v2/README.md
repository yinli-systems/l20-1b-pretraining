# Mixture and resume qualification source v2

This local candidate adds exact prediction-token mixing and preserves CPT origin
on resume. It has not been deployed or qualified on CUDA/NCCL. It does not replace
the immutable formal base, `posttrain/train_cpt.py`, or the frozen remote source.

## Data behavior

`mixture.py` requires a pinned manifest and verifies the SHA-256 of every listed
NumPy shard at startup on every rank. Files must be one-dimensional uint16 arrays
of complete `sequence_length + 1` blocks. Each block contributes exactly
`sequence_length` prediction tokens. All positive source weights must exist;
missing/exhausted sources cannot silently change the mixture.

Integer percentage weights form the smallest exact period (20 blocks for M3).
Each period gets a deterministic SHA-256 permutation of its source slots. A final
partial period uses Hamilton rounding with source-id tie-breaking; the final
quotas match the research planner exactly. Source cursors advance sequentially,
with a seeded cyclic physical offset per epoch. This bounds working memory to
64 periods and preserves locality within each source. It is not document-level
random shuffling, a quality filter or a learned mixture policy. Source documents
must be appropriately shuffled/stratified before packing.

Global block index is `microstep * world_size * microbatch + rank * microbatch +
local_index`. Recreating the reader at the checkpoint's next index recreates the
same stream. Different ranks receive distinct logical occurrences. Explicit
replay can repeat physical documents; it is bounded by each source's declared
`(prior_prediction_tokens + planned_prediction_tokens) / unique_prediction_tokens`
cap. Previously consumed tokens must be attributed to this exact admitted pool.
Cross-source near duplicates are handled by corpus admission, not shard hashes.

The manifest schema is `p529m-packed-mixture-v2`. Required top-level fields are
`sequence_length`, `tokenizer_sha256`, `weights_percent`, and `sources`. Each
positive source requires `id`, a fixed 40/64-hex `revision`,
`prior_prediction_tokens`, `max_cumulative_epochs`, and a nonempty `shards` list.
Each shard supplies `path` (relative to the manifest or absolute), `sha256`, and
`blocks`. Source and shard order do not affect stream identity; shard content does.
Weights currently operate on the seven top-level corpus groups. The prescribed
math subsource and code/language submixtures still require preparation and their
own measured admission receipts; arbitrary concatenation is not sufficient.

## Recovery behavior

`resume.py` stores the original base SHA-256 in every v2 checkpoint and includes
it in the fingerprint on both fresh and resumed launches. Code hashes, runtime,
GPU type, topology, seed, full token budget, schedules, protocol, data identity,
admission receipt and validation identity are also bound. Changing these requires
a separate experiment. The checkpoint additionally stores the exact next global
block and every source's consumed-block count. Missing legacy origin is rejected;
legacy optimizer checkpoints are not silently reinterpreted. The existing formal
base may still initialize a fresh CPT run using only its model weights.

Model, AdamW, Python, NumPy and Torch CPU/CUDA RNG states are saved. Resume checks
the checkpoint sidecar hash before deserialization and restores RNG after model
and DDP construction. Data must remain immutable during a launch. The checkpoint
file and hash sidecar are individually replaced; a crash between the replacements
will be rejected, not automatically repaired. This is an integrity check, not a
power-loss recovery guarantee. Full GPU restart qualification remains required.

## Qualification runner

`train_cpt.py` integrates both modules into a copy of the old DDP trainer, with a
mandatory 1–512-step qualification limit per launch. The model architecture is
unchanged. `P529M_MODEL_SOURCE` locates the pinned `model.py`; it defaults to the
local `pretrain500m` directory. Run `python3 train_cpt.py --help` for required
manifest, protocol, initialization, validation, and admission arguments.

Before GPU setup the runner requires a hashed `p529m-corpus-admission-v1` receipt:
`status: PASS`; matching `training_manifest_sha256`,
`validation_manifest_sha256`, and `protocol_sha256`; and `PASS` for checks
`licenses`, `content_quality`, `cross_source_deduplication`,
`benchmark_decontamination`, and `family_disjoint_splits`. This consumer verifies
receipt identity and status. The actual audit and supporting evidence are still
to be built; writing PASS labels is not an audit.

The runner checks disk headroom, exact batch divisibility, training/validation
shard separation and absence of validation replay. It saves a full final
checkpoint and reports `QUALIFICATION_COMPLETED`, never formal promotion. A
completion with too few steady-state steps reports `mfu_window_passed: false`.
Real qualification must demonstrate sustained rolling median MFU strictly above
0.50 using the correct dense BF16 GPU denominator. Single aggregate validation
loss here is a plumbing check; five-domain development/confirmation evaluation
and generated correctness checks remain separate unfinished work.

## Local checks and next work

From the workspace root:

```sh
python3 -m pytest -q pretrain500m/posttrain/v2/test_v2.py
python3 pretrain500m/posttrain/v2/train_cpt.py --help
```

Tests cover quotas at partial/full periods, all prefix cursors, 1/2/4/8-rank batch
assignment, reader recreation, seed/source-order behavior, bounded cache,
missing/corrupt/duplicate data, dtype/shape failures, replay caps, invalid inputs,
all four real screen quotas, and exact continuous-versus-resumed CPU training
with AdamW, dropout and Python/NumPy/Torch randomness. These tests use small
synthetic arrays and a tiny CPU network, not the 529M GPU model.

Next: admitted diverse corpus and submixtures; family-disjoint multi-domain
development sets; a working transport route; freeze this candidate and execute
paired continuous/resumed GPU runs with the same full LR schedule; measure
actual mixture MFU; only then enable the planned screen and confirmation stages.
