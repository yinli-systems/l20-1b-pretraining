# Faster quality checks with unchanged model outputs

**Fixed-model evaluation throughput: 1.2983x** in the final fresh-process test. This is an engineering result, **not a new trained model or accuracy gain**.

## Remove redundant work rather than change arithmetic

The original local diagnostic microbatch performs correct-image generation, deterministic wrong-image generation, and content/EOS NLL. It decodes/processes the same images and runs the frozen vision path separately for each check. `PreparedEvaluation` creates one scoped, immutable-image bundle and reuses those features for the three checks. It retains the existing image ordering, masking, BF16 decoder, IEEE-FP32 high vision, greedy generation, and scoring semantics. There is no cross-batch cache and no optimizer.

## Measured end-to-end result

| Metric | Original | Prepared bundle |
|---|---:|---:|
| Mean time / 64 images | 18.0674 s | 13.9162 s |
| Relative evaluation throughput | 1.0000x | **1.2983x** |
| Wall time reduction | — | **22.98%** |

Timing includes image reading, preprocessing, bundle creation, both generation passes and NLL. Each mode has four measurements in two ABBA blocks over the same 64 real TRAIN images. The block ratios are 1.2990x and 1.2976x. Peak allocated memory increased by 8.54 MiB in the measured process. These are small repeated measurements, not training-seed statistics or a guaranteed service-level gain.

**Scope matters:** this does not speed every component of training by that ratio, and it does not establish a gain for the single-generation public-benchmark pass. It optimizes the existing three-part local quality diagnostic. No reduced-precision or compiled vision candidate from the previous session was silently enabled.

## Numerical and state checks

The standalone final program reloaded the immutable step-4608 parent in a **fresh process** and checked:

- 128 real T448 TRAIN images, 64 PlotQA and 64 single-page Docmatix.
- **All 256 correct/wrong-image generated token sequences identical**, not merely normalized answer strings.
- Every recorded content NLL, EOS NLL, first-content-token accuracy and token count identical.
- Original224 regression on eight natural images: both generation modes and diagnostic values identical.
- Original trainable-parameter fingerprint unchanged; zero optimizer updates.

The initial session additionally tested thread/stream ownership, use-after-close, duplicate-image controls, mask shape and restoration after exceptions. It observed 1.2808x on the T448 workload before a low-route regression was found. The final measurements above come from the corrected implementation, not that initial session.

## Failed candidate regression retained

V1 passed T448 but **failed the original224 check**, with a largest diagnostic difference of 0.0101636 and changed generated tokens. The prepared feature computation had been moved outside the original outer BF16 autocast context. T448 disables autocast internally, so it was unaffected; the original224 route does not.

V2 restores the exact outer autocast scope before precomputing features. The final fresh-process run repeated both the full128 T448 check and the original224 regression successfully. This was a bug in the optimization candidate, **not an asserted defect in the historical trained model**. V1 source, failed diagnostic and correction record remain included. In the initial `summary.json`, its original source names refer to the corresponding archived `*-v1.py` files; the final source hashes are pinned by `fresh-v2/policy.json`.

## Reproduction

GPU execution requires the user's existing local model and admitted images. The program is evaluation-only and uses a separate output directory:

```bash
python qualify_prepared_eval.py \
  --workspace /path/to/vision-scale50-workspace \
  --output /path/to/new-evaluation-output \
  --images 128 --timing-images 64 --abba-blocks 2
```

The workspace must contain the pinned `program.json`, `spatial_route.py`, its existing loader dependencies, the healthy checkpoint, and the hash-bound reconciled data. No assets are downloaded automatically. A model/row bundle is single-thread, single-stream and scoped to an unchanged model. Arbitrary `.data` mutation, concurrent model edits, or reuse across optimizer steps are outside this evaluator contract. H896 has not been qualified by this final evaluator test.

CPU evidence verification needs no GPU:

```bash
python verify.py
```

## What did not happen

No new trainer was created or executed; the previously blocked trainer-creation action was not retried. No new image corpus was bulk-downloaded and no checkpoint/data was deleted. The 8.4M unique-image target, old56.25% retention anchor and51.25% floor remain unchanged. Model-quality improvement still needs an approved training/quality experiment. The released language model and public benchmark scores are unchanged.

## Implementation references

- [Transformers4.56.2 generation implementation](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/utils.py): the installed generation already supports last-token logits and KV caching; these were not presented as newly added optimizations.
- [PyTorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html): profiling guided the investigation; the timing result above was measured outside the profiler.
- [PyTorch automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html): preserve the original autocast context when relocating computation.
