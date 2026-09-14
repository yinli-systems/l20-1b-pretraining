# Fifth-tranche fast pipeline

This fail-closed runner waits for the active intake writer, verifies all sixteen
new segments, removes only the stale writer lock, and launches the tokenizer
audit, exact-span contamination scan, and legacy-hash scan concurrently. It
reuses frozen four-tranche evidence by hash and scans only the appended tranche.
Old-corpus overlap, exclusion union, quality features, family closure, and
packing remain ordered because each consumes the preceding bound report.

Completion creates a hash-bound receipt with `training_admitted=false`; quota,
manifest, held-out, epoch-cap, and live-MFU gates are still required before a
training job can start.
