# Bounded raw corpus acquisition

The worker fetches revision-pinned Parquet row groups directly on the training
site through `hf-mirror.com`, using `?download=true`, verified HTTP byte ranges,
the expected upstream file size, and recorded range hashes. Fixed source revisions
and expected LFS SHA-256 values were obtained from the primary Hugging Face API.
Partial row-group acquisition does not verify the full-file LFS SHA-256; receipts
explicitly record that distinction. Raw metadata is retained with physical row
positions. No dataset-supplied code is executed.

The first tranche covers 16 source/language strata, at most four row groups per
stratum. Limits are 128 MiB of Parquet network reads and 192 MiB of uncompressed
output per stratum, 50,000 text rows, and 5,000 attempted code rehydrations.
Two stratum workers run concurrently; code uses at most eight fetchers within
each worker. Acquisition pauses at the next group if free space falls below
18 GiB. S3 code bodies have bounded decompression, raw-content SHA-1 matching,
UTF-8 decoding, known permissive-license filtering, and Python AST parsing.
Syntax checks do not establish correctness. S3 network traffic is separate from
the Parquet range budget and is bounded by the code attempt/body limits.

The code archive SHA-256 is
`e29e1497921ef3fc0426d11290cd1e075ced1a8faeede6ecfa0447c3da4c5e16`.
The original worker PID is 2559932 on the login node; verify its identity before
acting on that PID. Source, logs and outputs are respectively:

- `/ssd/scxi253/pretrain500m-20260912-v1/source/intake-v1`
- `/ssd/scxi253/pretrain500m-20260912-v1/logs/intake-v1.log`
- `/ssd/scxi253/pretrain500m-20260912-v1/data/diverse-intake-v1`

Existing incomplete outputs require inspection; the worker refuses to overwrite
them silently. A writer lock prevents a second writer. Files marked
`RAW_SEGMENT_READY` are raw inputs, not admitted training shards. Four row groups
per source are a first tranche, not enough by definition for the planned screen.
Measure actual retained tokens, document families and each leaf's quota shortfall
before extending acquisition to additional, disjoint groups.

As of 18:34 UTC, 4,000 FineMath and 4,000 InfiWebMath documents had landed,
Java code was also present, and other code/language strata were still processing.
Cross-source deduplication, benchmark decontamination, family splitting, language
and factual quality checks, packing, and actual quota admission remain pending.
