# F2 continuation raw intake

The frozen plan acquires 31 unused DCLM row groups, five unused English PDF
groups, and four unused C++ groups from revision- and content-pinned shards.
It uses the old intake reader with exact code hashes, historical receipt/output
verification, deterministic row-group exclusions, finite byte and row budgets,
and a single-writer lock. The C++ path accepts only verified permissively
licensed code blobs. The output remains raw and **not training admitted** until
family, rights, quality, contamination, tokenizer pack, and lineage-cap gates
pass. This is an intake for the 0.537B-token-per-seed F2 continuation pilot;
it does not launch training or qualify the resulting model.
