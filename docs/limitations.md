# Claims and limitations

## What the latest result supports

Under the fixed custom-wrapper evaluation, the streamed endpoint improves
TextVQA and DocVQA over its starting checkpoint. This is a single-run result
with paired image-cluster confidence intervals. Full denominators and stored
paired sufficient counts are in the [result package](../results/2026-09-28/README.md).

## What it does not support

It is not a frontier model, a SmolVLM-parity result, a pure data-quantity ablation
or a population-level scaling law. Source mixture, replay and multiple-choice
label formatting changed together. Higher-resolution or trainable-vision
variants remain hypotheses for controlled tests.

**Retention failed.** Old natural-image QA fell from 56.25% to 50.86%; two
consecutive checks exceeded the 5-percentage-point tolerance. Other retention
checks passing does not erase that failure. The final model is not promoted.

AI2D's baseline default-filter score is 7.32; the same original predictions
score 25.62 under a separate prefix-tolerant diagnostic. That diagnostic is not
a replacement for the primary metric. The 26.26 after-score therefore must not
be presented as a 19-point gain in visual reasoning alone.

ChartQA's 95% paired interval crosses zero. Two known exact-overlap questions
from prior training remain in the reported full split; all questions are kept
and the earlier sensitivity result is separately recorded.

## Statistics and evaluation

The four full published splits contain 15,937 questions. They were evaluated
with a custom wrapper aligned to pinned scoring/task definitions, not the full
lmms-eval CLI. No leaderboard submission is claimed. Public baseline results
were inspected before the new run and informed format handling, so they are
not a blind confirmatory endpoint. Intervals describe evaluation-image
sampling only, not training seeds, search choices or four-task multiplicity.

## Data, code and release

Exact hashes, pixel hashes and registered near-image checks are useful but do
not certify semantic or original-document independence. Prepared source counts
are not completed optimizer exposures. Final runtime in the saved training
summary covers the resumed process only; it is not the entire run duration.

Published evidence uses machine-path aliases. Both original-source and
published-file hashes are recorded; transformed JSON is not falsely claimed
byte-identical to the remote files. Paired score exports contain image hashes,
question counts and score sums, not images, prompts or reference answers.

The text base is not instruction- or safety-tuned. VLM weights and optimizer
states remain private. The public repository is not a self-contained current
VLM release. Code MIT licensing does not relicense external models or datasets.
