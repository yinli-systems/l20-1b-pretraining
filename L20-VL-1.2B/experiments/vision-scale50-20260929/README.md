# Vision scale-up: data repair and real-L20 qualification

**Status: data contract repaired; bounded GPU qualification complete; no new training updates.**

The healthy step-4608 parent remains unchanged. The 8.4M-unique-image goal is a target, not completed acquisition or training. This record includes unsuccessful optimizations rather than promoting every faster kernel.

[Machine-readable summary](metrics-summary.json)

## Follow-up: original evidence recovered; faster quality checks

The earlier transfer timeout is resolved. The **26-file original execution
package** was recovered with matching remote/Mac archive hashes and its
published manifest verified. Thirteen recorded headline cross-checks agree
with the earlier transcribed summary. See [recovery receipt](export-recovery-20260929.json)
and [original evidence](original-execution/README.md). The original files are
preserved; the operational-boundary paragraph below describes the earlier state.

The new [prepared-evaluation module](prepared-evaluation/README.md) measured
**1.2983x throughput / 22.98% less wall time** for the local
true-image / wrong-image / content-NLL bundle, including reading and preprocessing.
A fresh-process qualification reproduced all256 T448 generated sequences on128
real images and passed the original224 regression after correcting an AMP-context
bug in the first candidate. The failed candidate remains recorded.

This is **not** a training-throughput, public-benchmark single-pass, or model-quality
improvement. No optimizer steps occurred. Main model scores and the8.4M-image
target are unchanged. [Latest measured summary](prepared-evaluation/headline.json).


## Completed data repair

The previous pilot JSON and SQLite registry disagreed on **1,968 of 9,386 records**. The repair creates a separate derived corpus and preserves the union of both original holdouts. An image is training data only when both old registries classify it as training. Conflicting development/confirmation assignments are excluded; no heldout image moves into training.

| Derived subset | PlotQA | Single-page Docmatix | Total |
|---|---:|---:|---:|
| Training | 6,239 | 1,161 | **7,400** |
| Development | 800 | 163 | 963 |
| Confirmation | 843 | 161 | 1,004 |
| Excluded | 18 | 1 | 19 |

All **9,386 image files** passed their recorded SHA-256 checks. The new row manifest and ledger agree on every split and document key. Seventeen conflicting heldout assignments and two pre-identified TRAIN quality concerns are excluded. The original files remain unchanged.

Trailing punctuation was normalized on **1,186 complete numeric TRAIN labels**, including scientific notation. Heldout references and benchmark scorers were not changed. The actual-pilot integration and unit suite passed **26 tests**, including negative checks for altered heldout references and split assignments. This is not an exhaustive semantic annotation audit.

## GPU candidates and promotion decisions

All numerical probes use **64 fixed real TRAIN images**, 32 per source, with no public test or pilot holdout images. Greedy agreement below means agreement with the reference implementation, **not task accuracy**.

| Candidate | Greedy agreement | Largest per-example answer NLL difference | Outcome |
|---|---:|---:|---|
| T448 + TF32 high vision | 60/64 | 0.090451 | Failed predeclared numerical gates |
| T448 + BF16 high vision | 58/64 | 0.063206 | Failed predeclared numerical gates |
| H896 + compiled FP32 high vision | 61/64 | 0.031030 | Feature path 1.0873x, but numerical gates failed |
| H896 + CUDA Graph, original arithmetic | 64/64 | **0.0** | Feature path **0.9905x**: no measured speed win |

The T448 precision probes and H896 implementation probes have different references. H896 still uses an **untrained initial packing projection**. None of these results is evidence that H896 improves the model's accuracy.

The reduced-precision gates were recorded before testing: feature relative L2 <= 0.01, per-example answer NLL difference <= 0.02, and greedy agreement >= 0.98. Changed predictions do not necessarily mean worse predictions; these candidates would require separate training/quality qualification instead of being called equivalent drop-in acceleration.

Increasing H896 native-tile batches from 16 to 32/64/128/256 produced approximately 1.08-1.10x feature-path ratios, but relative feature differences of 1.43e-5 failed that study's predeclared 1e-6 gate. No tolerance was relaxed afterward. Performance repetitions were small and concern the feature path only, not complete training throughput or serving performance.

### CUDA Graph numerical and ownership checks

The original-arithmetic executor matched **184 trainable-tensor gradients** on a two-image backward test: relative gradient L2 difference 0, cosine 1, identical loss, finite projection gradients, and no frozen-encoder gradients. There were **zero optimizer steps**. The trainable projection remains outside the captured/no-gradient encoder region.

Six guards passed: independently owned output storage, no overwrite of earlier outputs, exact A-B-A replay, and rejection of wrong input shapes, other CUDA streams, and other threads. Arbitrary external `.data` mutation is outside the immutable-encoder contract.

The first graph-harness attempt incorrectly applied the projection outside the IEEE-FP32 precision context. That failed diagnostic was retained and excluded from performance evidence. After correcting the context, feature, NLL and gradient equality passed. Since measured speed did not improve, the graph is **not enabled by default**.

## Input-runtime stress test

The bounded thread prefetcher performed **3,000 actual image reads, SHA checks and PIL decodes**, verifying order on every item. It used **64 distinct images**, not 3,000 new training examples. There were two workers and at most three pending items. File-descriptor counts were **40 before and after** and 40 at every 250-item sample; active threads returned from six during work to the original four afterward.

This provides bounded stress evidence, not proof of multi-day or 8.4M-image stability. Only 49/1,024 native tiles in the probe were uniform, so a blank-tile cache was not implemented.

## Next quality experiment, not yet executed

The next admission stage should compare the same healthy parent and reconciled training rows under matched image/update budgets:

1. Original C448 processing.
2. Aspect-ratio-preserving 448 processing, isolating letterboxing from resolution.
3. H896 processing and trainable packing at the same 784-token language budget.

Keep the original **56.25% old-QA anchor and 51.25% stop floor**. Fix candidate selection and retention rules before opening final confirmation results. Report both equal-update comparisons and actual wall time; higher resolution may be more expensive despite an unchanged language-token count. Only then decide which recipe should progress toward 8.4M unique images. Do not turn repeated images, question count, or text-only rows into an apparent 50x increase.

## Operational and publication boundary

The previous remote trainer-creation action was safety-blocked. It was not retried or routed through another mechanism in this work. The completed actions are separate data repairs, forward tests, no-update gradient checks and input-runtime tests. **No new trainer, trained checkpoint, or model-quality improvement is claimed.**

The original checkpoint members were hash-checked again and the old trainable-parameter fingerprint was unchanged. The last confirmed L20 status was **0 MiB / 0% at 2026-09-29 05:27:18 UTC**. Later Desktop Commander response and ping calls timed out; the directory's online label alone does not establish working terminal connectivity.

Full source/receipt export readback is pending because that connection timed out. The public JSON is explicitly **transcribed from completed tool-returned measurements**, not represented as a byte-identical raw receipt export. No raw datasets, image bytes, weights, optimizer states or private predictions are included. The main README and released model metrics remain unchanged.

## Primary references

- [SmolVLM](https://arxiv.org/abs/2504.05299): motivates controlled image-splitting and token-compression experiments, not guaranteed gains for this model.
- [MM1](https://arxiv.org/abs/2403.09611): motivates separate treatment of resolution, token count and data mixture.
- [FineVision](https://arxiv.org/abs/2510.17269): supports careful data-curation and benchmark-overlap checks instead of assuming a dataset name or quality threshold guarantees improvement.
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html): mathematically equivalent batched operations need not be bitwise identical.
- [PyTorch compile](https://docs.pytorch.org/docs/stable/generated/torch.compile): compilation modes and overhead require measurement in the actual workload.
