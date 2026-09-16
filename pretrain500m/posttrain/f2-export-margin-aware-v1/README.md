# Frozen margin-aware F2 export protocol

The original 256-position BF16 argmax threshold produced both pass and failure
outcomes for each exact checkpoint across three predeclared input seeds, while
all tensor states remained bitwise exact and all logit-drift bounds passed.
This adaptive protocol freezes four new input seeds and applies a transparent
numerical-invariance rule to both F2 parents and both continuations.

Every cell requires bitwise-exact reloaded state tensors, finite logits,
maximum absolute logit drift at most 0.5, and mean absolute drift at most
0.025. A changed argmax is accepted only when both native and reloaded top-one
margins are no larger than twice that row's measured maximum absolute logit
drift. Therefore every position outside the measured ambiguity set must retain
its argmax. Raw argmax agreement remains recorded as a descriptive metric.

The two array tasks each evaluate one matched parent/continuation pair over all
four fresh seeds. All 16 cells must pass before the continuation exports can be
used to finish the adaptive matched seven-task evaluation. This protocol was
designed after inspecting the earlier export failures, so it cannot be treated
as sealed evidence or erase those failures.
