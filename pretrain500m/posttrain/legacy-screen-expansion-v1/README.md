# Independent legacy scan

This required check runs alongside the tokenizer and supplemental Unicode scans.
It does not wait for their row indexes or retokenize documents. Two Linux fork
workers share the immutable legacy hash table and process larger files first.
Each worker owns separate output files. The parent validates all original input
receipts with the frozen raw-audit-v2 preflight and rechecks hashes/file sets after
completion. The original benchmark SQLite database is opened read-only.

The scanner preserves the original ASCII 13-word BLAKE2b-64 membership algorithm.
Matches are candidates because this database stores hashes rather than reference
spans. This does not replace the supplemental Unicode/code scan or establish
semantic cleanliness. It also writes the original packer's NFKC/lower/whitespace
document hashes, allowing the old-FineWeb comparison to reuse this pass. That
comparison, family expansion, final filtering and admission are separate work.

The fixture suite checks parity against the actual original packer's two pure
functions, including case, whitespace, the 13-word boundary, and lower versus
casefold. No raw data, evaluation content or frozen auditor is modified.

This expansion copy adds `--scan-from-tranche`. It verifies all historical raw
bindings and physical group history, then computes candidate hashes only for
the appended tranches. The resulting candidates must extend the frozen prefix
exclusion union before quality filtering.
