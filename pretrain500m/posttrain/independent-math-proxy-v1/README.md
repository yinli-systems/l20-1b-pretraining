# Independent math proxy v1: MGSM English direct answer

This is the math component of the new development proxy. It uses the pinned
original MGSM English TSV that was already held outside training. The 250 rows
are split by a frozen output-blind SHA-256 ordering into 125 development rows
and 125 confirmation rows. Only development may be scored during mixture work.

The primary metric is greedy direct-answer numeric exact match. The prompt is
`Question: {question}\nAnswer:`; decoding stops at the first newline or after 64
new tokens, and the first numeric expression is compared by exact decimal
value. Mean negative log likelihood per gold answer token is a secondary
directional metric. It is not a substitute for generated-answer accuracy.

MGSM is public and may occur in model ancestors. The project exact-span scans
reduce known local intake leakage but cannot prove semantic non-contamination.
This development proxy is not sealed and cannot establish broad mathematical
reasoning or model-market superiority.

