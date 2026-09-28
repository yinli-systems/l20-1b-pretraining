# Streamed multimodal expansion

**September 28, 2026 · one L20 · 5,120 updates · 15,937 evaluation questions**

[Metrics](metrics.json) · [Source hashes](source-manifest.json) · [Aggregate evidence](evidence/) · [Paired score counts](scores/) · [Limitations](../../docs/limitations.md)

## Outcome

The fixed endpoint improves TextVQA and DocVQA under the recorded wrapper, but
fails old-QA retention. ChartQA's gain is unresolved and AI2D contains a major
format confound. **The endpoint was retained for analysis, not promoted.**

| Task | Questions / image clusters | Before | After | Change, pp | Paired 95% CI, pp |
|---|---:|---:|---:|---:|---:|
| TextVQA | 5,000 / 3,166 | 17.506 | 23.774 | +6.268 | [+5.346, +7.163] |
| DocVQA | 5,349 / 1,285 | 4.591 | 9.483 | +4.893 | [+4.223, +5.557] |
| ChartQA | 2,500 / 1,509 | 11.320 | 12.160 | +0.840 | [−0.160, +1.888] |
| AI2D | 3,088 / 814 | 7.319 | 26.263 | +18.944 | [+17.391, +20.604] |

Scores are displayed on a 0–100 scale. Metrics differ by task. All questions
are retained; no combined macro is used to obscure task-specific regressions.

## Retention and stopping

Old natural-image QA started at **56.25%**. It fell to **50.703% at update 4,864**
and **50.859% at update 5,120**. Both exceed the predeclared 5-point drop
allowance. Two consecutive failures triggered `retention_guard`.

Caption, text and the other recorded NLL checks passed, but that does not
cancel the old-QA failure. The pipeline's successful shutdown is not evidence
that every quality gate passed. See the [final health record](evidence/training/step-005120-health.json)
and [previous failed check](evidence/training/step-004864-health.json).

## What changed

The run starts from the previous 1,934-update endpoint, freezes language and
vision backbone weights and updates the 10.28M bridge/attention-LoRA parameters.
New sources are streamed once alongside old-task replay and text retention.
The replay mixture and multiple-choice target formatting also changed; this
is not a controlled data-size-only experiment.

| Accounting item | Value |
|---|---:|
| New-image events in the saved lineage | 163,840 |
| Prepared images | 165,376 |
| All image events | 327,680 |
| Supervised answer tokens | 3,350,198 |
| Text replay prediction tokens | 5,242,880 |
| Requested upper bound | 250,000 new images, not reached |
| Final checkpoint manifest | `ff8a6684ec1922a04e7948dbfbbfb8f345c64c650f6fa99ceb311ebdc6af11b0` |

A file-descriptor exhaustion stall occurred after update 3,856. The run resumed
from checkpoint 3,840; 16 uncheckpointed updates were replayed. The raised
open-file limit was a mitigation, not proof that the input lifecycle leak was
eliminated. The saved summary's 4,842.99 training seconds covers **only the
resumed process**, not total compute or elapsed time.

## Evaluation contract

Complete published validation/test splits were evaluated before/after with the
same rows, data hashes, task prompts, batch size 8 and greedy generation.
The source route is 448px → four native 224 tiles → 784 visual tokens.
No external OCR or gold boxes were supplied. TextVQA/DocVQA use 32 maximum new
tokens; ChartQA/AI2D use 16.

The task/scorer reference is pinned to lmms-eval commit
`42296f8fd2f0929d88a6ac4f7a1f2d784537f8d8`, but execution uses the project's
custom wrapper. It is neither the full harness CLI nor a leaderboard submission.
The paired bootstrap uses 4,000 image-cluster resamples. Intervals cover
sample uncertainty, not training-seed variance; one seed and four exploratory
domain comparisons do not establish a general scaling law.

## Confounds that remain visible

**AI2D:** before training, 2,240 predictions included `Answer:`. The default
primary score was 7.32; a prefix-tolerant rescore of those same predictions was
25.62. The latter is a diagnostic, not a replacement primary result. The large
primary increase cannot all be attributed to improved visual reasoning.

**ChartQA:** two known exact-overlap questions from prior training are included
and disclosed. Its paired interval crosses zero. Restricted CoSyn chart-title
transcription does not supply general chart-arithmetic supervision; the result
is not proof of an architecture ceiling.

**Benchmark reuse:** baseline diagnostics were inspected before training and
informed formatting changes. Public benchmark results are not a blind final
confirmation. Semantic/document-level overlap is not exhaustively excluded.

## Public verification and release boundary

```bash
python3 tools/verify_repository.py
```

Run from the repository root. Per-image CSVs contain question counts and
before/after score sums, allowing headline means to be independently recomputed
without redistributing prompts, labels or model-generated text. They do not
regenerate the predictions themselves. Published JSON is path-sanitized;
original and published hashes are distinguished in the source manifest.

No raw images, model weights, optimizer states or credentials were added to
Git. The language base remains a separate released model; this VLM remains a
research endpoint below the project's intended capability and retention goals.
