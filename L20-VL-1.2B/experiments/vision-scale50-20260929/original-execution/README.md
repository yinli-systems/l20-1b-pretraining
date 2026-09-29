# Vision scale-up: data repair and bounded GPU qualification

**Status: no new training updates; no new model-quality claim.** The healthy
step-4608 parent is unchanged. This package records completed data repairs and
real-L20 numerical/performance tests, including rejected candidates.

## Data contract repaired

The previous pilot JSON and SQLite registry disagreed on 1,968 of 9,386 rows.
The repair uses the union of both prior holdouts. A development/confirmation
conflict is excluded; no reserved image becomes training data. The originals
remain unchanged and the repaired corpus is a separate artifact.

| Repaired subset | PlotQA | Single-page Docmatix | Total |
|---|---:|---:|---:|
| Training | 6,239 | 1,161 | **7,400** |
| Development | 800 | 163 | 963 |
| Confirmation | 843 | 161 | 1,004 |
| Excluded | 18 | 1 | 19 |

All 9,386 image files passed their recorded SHA-256 checks. Both the new ledger
and rows agree on every split and document key. Seventeen conflicting heldout
assignments and two pre-identified TRAIN quality concerns remain excluded.
The repair normalizes trailing punctuation on **1,186 complete numeric TRAIN
labels**, including scientific notation. It does not edit heldout references or
any published benchmark scorer. This is not an exhaustive semantic-label audit.

The actual-pilot integration and unit suite passed **26 tests**. Negative tests
reject altered heldout labels, changed splits, and ledger mismatches. The source
script expects the private pilot directory and is not a dataset release.

## GPU candidates: numerical gates before promotion

| Candidate | Exact greedy agreement | Largest per-example NLL change | Timing/decision |
|---|---:|---:|---|
| T448 + TF32 | 60/64 | 0.090451 | Not drop-in equivalent under the frozen gates |
| T448 + BF16 vision | 58/64 | 0.063206 | Not drop-in equivalent under the frozen gates |
| H896 + compiled FP32 vision | 61/64 | 0.031030 | Feature path 1.0873x; not admitted |
| H896 + CUDA Graph, original arithmetic | 64/64 | **0.0** | Feature path **0.9905x**; no speed win |

T448 precision and H896 executor studies have different references. H896 still
uses its **untrained initial packing projection**; agreement is a numerical
qualification, not benchmark accuracy. An answer change does not prove the new
answer is less accurate. Reduced precision would need a separately qualified
training/quality recipe rather than being called an equivalent acceleration.

Increasing H896's native-tile batch from 16 to 32/64/128/256 gave roughly
1.08-1.10x feature-path ratios, but relative feature differences of 1.43e-5
failed that study's predeclared 1e-6 gate. No threshold was relaxed afterward.
All timing comparisons retain raw ABBA records; these small measurements do
not establish population-level speedups or complete training throughput.

The CUDA Graph implementation matched **184 trainable-tensor gradients** on
a real two-image backward test: gradient relative L2 difference 0, cosine 1,
finite gradients, and no frozen-encoder gradients. There were **zero optimizer
steps**. Its packing projection remains outside the cached/no-gradient region.

Six executor guards passed: independent output storage, no overwrite of an
older result, exact A-B-A replay, and refusal of wrong shapes, other CUDA
streams and other threads. Arbitrary external `.data` mutation is explicitly
outside this immutable-encoder contract. The first graph-harness attempt had
a projection precision-context bug; its failed diagnostic is retained instead
of being counted as a benchmark.

## Input-runtime stress test

The existing bounded thread prefetcher read, hash-checked and decoded actual
images **3,000 times**, verified output order, and returned to **40 open file
descriptors and 4 threads**, matching its starting counts. Pending work stayed
at 3. Only **64 distinct images** were used: this is an I/O stress test, not a
3,000-image scale-up or proof of multi-day stability. Uniform tiles were just
49/1,024 in this probe, so a blank-tile cache was not implemented.

## What remains before the 50x run

The goal remains **8,400,000 unique new images**, not repeated questions or
multiple epochs. No 8.4M-image acquisition or training completion is claimed.
The next quality experiment must use the reconciled data, the unchanged
healthy parent, and matched image/update budgets. Preserve the original
56.25% old-QA anchor and 51.25% stop floor. Include an aspect-ratio control so
higher resolution, letterboxing and learned packing are not conflated.

The previous remote trainer-creation action was safety-blocked. It was not
retried or routed through another mechanism here. This turn performed separate
data repair, read-only forward tests and no-update gradient qualification.
There is no new trainer, no new trained checkpoint, and no automatic promotion.

## Verify the public evidence

```bash
python L20-VL-1.2B/experiments/vision-scale50-20260929/verify.py
```

This checks the published files and aggregate contracts without a GPU. It does
not regenerate model predictions. The GPU measurements were executed in an
interactive Python session on the L20; reusable graph/data modules are included,
but this package is not a one-command model or training reproduction release.
The earlier public quality scores and the released language model are unchanged.

## Primary research and implementation references

- [SmolVLM](https://arxiv.org/abs/2504.05299): image splitting and space-to-depth motivate the H896 experiment, not a guaranteed gain.
- [FineVision](https://arxiv.org/abs/2510.17269): quality-score thresholds and cross-benchmark overlap need separate scrutiny.
- [MM1](https://arxiv.org/abs/2403.09611): resolution, visual tokens and data mixture warrant controlled ablations.
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html): mathematically equivalent batches need not be bitwise equal.
- [PyTorch compile](https://docs.pytorch.org/docs/stable/generated/torch.compile): compilation overhead and actual end-to-end gains must be measured.
- [PyTorch DataLoader issue 91252](https://github.com/pytorch/pytorch/issues/91252): repeated persistent pinned loaders motivated the bounded-thread prototype; this is not proof that every upstream issue is fixed.
