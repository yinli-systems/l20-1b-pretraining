# Parallel comparison against the old source snapshot

Up to sixteen CPU workers scan the frozen original shared FineWeb-Edu shards read-only.
Every full Parquet file is checked against its expected pinned LFS SHA256 before
scanning, and file identity/size/mtime are checked after reading. Workers stream
512 rows at a time and own separate output files. The immutable new-document
normalization indexes are reused from the completed legacy scan, with hashes and
row counts checked. No original corpus copies or tokenized copies are made.

Comparison preserves the original packer's NFKC/lower/whitespace SHA256 semantics.
Both raw hashes and matched physical rows are retained, so normalized matches are
not mislabeled as byte-identical documents or equivalent programs. Matches against
the complete original source snapshot do not prove membership in the final pack;
the original language/quality eligibility is recorded separately. Host counts are
metadata for subsequent family checks, not independent-family certification.
Near duplicates and semantic/translation overlap remain separate requirements.

Three owned-fixture tests verify real Parquet reading, original filter scope,
physical row identity, original case normalization and changed-file rejection.
