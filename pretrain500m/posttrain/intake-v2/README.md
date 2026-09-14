# Additional raw data chosen from measured shortages

This separate acquisition extends ten deficient sources from intake-v1. It uses
the same pinned upstream file revisions, strict bounded HTTP ranges, metadata
preservation, permissive-code license filters and S3 content identity checks.
Each source verifies the prior receipt hash and excludes every already-read
physical row group, including any previously only partially retained group.
The previous outputs are never overwritten.

Group requests are derived from actual token upper bounds measured with the
formal tokenizer, the lower-bound shortage for the largest single screen quota
at two epochs, and a 1.35 planning margin. They do not certify cumulative
screening/confirmation sufficiency. Limits remain 128 MiB of Parquet reads and
192 MiB uncompressed output per source, two source workers, eight S3 fetchers
per active code source, and 18 GiB free disk. Code permits at most 25,000 fetch
attempts per stratum; text permits 100,000 accepted rows. Per-source body/range
size limits and full-file-versus-partial-hash distinctions are unchanged.

The planned ten-source output ceiling is 1,920 MiB before gzip compression.
Existing sufficient non-Chinese language strata are not fetched again in this
tranche. Cosmopedia expansion is deferred until parent-document identity is
resolved; its `seed_data` source label is insufficient for that purpose.

Outputs under `data/diverse-intake-v2` remain raw and unadmitted. Hash verification,
token counts, deduplication, family separation, quality checks, contamination
checks and packing must include both tranches before any training admission.
