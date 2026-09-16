# Independent SciQ knowledge proxy v1

This protocol measures closed-book English science answer-choice likelihood on
the pinned SciQ test set. Before scoring, it excludes 12 rows detected by the
frozen current-corpus exact/lexical-near scan and nine duplicate-choice rows.
The remaining 979 rows are assigned output-blind to 489 development and 490
confirmation rows. Confirmation remains unscored.

The prompt contains only the question and `Answer:`. The `support` field is not
used. Choice ordering is a deterministic hash of ordinal and choice text and
does not use correct-answer identity. Primary accuracy selects the answer with
the highest mean answer-token conditional log likelihood; raw total likelihood
accuracy is secondary.

Three independent one-GPU Slurm array tasks score Base and the two long-F2
parents. Per-example results remain on ParaCloud. Only the compact summary,
logs, GPU receipts, and hashes may be archived in Git.

This is a public development proxy. Its contamination gate covers the current
F2 corpus with exact and fixed lexical-near rules; it does not establish
semantic-paraphrase or model-ancestor independence and cannot support broad or
market-superiority claims.
