# Separate first screen without unverified synthetic parents

This amendment responds to the user's request to accelerate execution, before any
new-mixture candidate result. It transfers the original 5% synthetic share to the
existing FineWeb-Edu replay source. All other source weights, two training seeds,
budgets, held-out requirements, MFU threshold and correctness checks are unchanged.
The original protocol remains intact for a later, separate synthetic-source ablation.
This is not an equivalent implementation of the original M-series recipes.

| Recipe | FineWeb | DCLM | PDF | Math | Code | Synthetic | Multilingual |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0 | 100 | 0 | 0 | 0 | 0 | 0 | 0 |
| F1 | 35 | 20 | 20 | 15 | 10 | 0 | 0 |
| F2 | 25 | 15 | 25 | 20 | 15 | 0 | 0 |
| F3 | 25 | 15 | 20 | 15 | 15 | 0 | 10 |

The largest FineWeb replay requirement is still set by the 100% control. No new
non-web leaf has a larger quota. Old/new-corpus overlap, cumulative per-run replay
accounting and independent families still require verification. Removing the
synthetic source removes its parent investigation from this screen's critical
path; it does not claim that the remaining corpus is admitted.

The independent legacy benchmark scan now runs beside token and Unicode scans.
Its old-normalization index can feed the old-FineWeb comparison without another
normalization pass over the new documents. Additional GPU concurrency requires
aggregate disk accounting: independent per-job free-space checks can all pass
while combined checkpoint writes exceed remaining capacity.
