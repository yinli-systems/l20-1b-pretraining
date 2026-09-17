# Negative results and engineering failures

These results are retained because scheduler acceptance, allocation, runtime,
quality, and scientific validation are separate evidence classes.

- Job 1595099: native `grouped_mm` preflight rejected the documented weight
  layout on torch 2.11; the installed kernel requires contiguous `[G,K,N]`.
- Job 1595108: whole-model compile failed before metrics on an FP32/BF16 fake
  tensor contraction.  Explicit BF16 grouped inputs/weights fixed the issue.
- Job 1595119: the first CVCR proxy attempt failed before metrics because a
  FP32 probe buffer received BF16 outputs.
- Job 1595128: the first valid CVCR path measured 46.44% wall overhead and was
  rejected.
- Job 1595140: sharing one uniform probe expert removed fragmented groups, but
  20.00% overhead still failed the hard systems gate.
- Job 1595144: whole-model compile exposed a scatter source/destination dtype
  mismatch in the custom backward; no training evidence was produced.
- Jobs 1595163/1595164: fixed sampling without replacement failed closed during
  early router collapse because a sampled expert had fewer inactive rows than
  the fixed probe budget.  Fixed-shape sampling with replacement and exact
  duplicate-aware inclusion probabilities replaced it.
- Job 1595176: a profile attempted to use the quota-limited home Triton cache;
  it failed before the CVCR profile.  All profile and training scripts now use
  exact node-local cache directories.
- Jobs 1595206/1595207: `reduce-overhead` whole-model compilation failed; the
  default compile mode remained faster and stable.
- Job 1595547: compiling the DDP wrapper instead of compiling the raw model
  measured 434,934 versus 434,423 tok/s after warm-up (+0.12%).  This is
  noise-sized and missed the pre-registered 5% promotion threshold, so the
  existing compile-before-DDP order remains the default.
- Job 1595562: DDP bucket caps of 25, 8 and 100 MiB measured 435,188,
  433,371 and 425,789 tok/s respectively.  Neither alternative beat the
  existing 25 MiB default, and 100 MiB regressed throughput by 2.16%.
- Job 1595576: fused-loss microbatch 32 fit and completed, but microbatch 64
  failed consistently on all four RTX 4090 ranks.  Each rank had only about
  377 MiB free when a further 384 MiB allocation was requested, so mb64/acc1
  is outside the measured memory frontier.
- Job 1595582: the first mb32 one-update parity harness ran both models eager
  and OOMed while Liger requested a 148 MiB gradient buffer.  Compiled 4-GPU
  mb32 training had already completed with about 2 GB more headroom, so this
  is retained as a mismatched-harness failure and superseded by a compiled
  parity rerun.
- Job 1595521: the pinned ScatterMoE backend reproduced a 12.24%--12.53%
  throughput gain in all three 381-step seed pairs, but failed the frozen
  validation non-inferiority gate.  Mean validation CE delta was +0.01549 and
  the one-sided 95% upper bound was +0.14655 versus the +0.002 margin; the
  backend therefore remains a performance candidate rather than a promoted
  default.
- Job 1595607: ScatterMoE plus Liger at microbatch 32 reproduced a
  28.62%--31.35% throughput gain across the three sustained seed pairs and
  improved mean validation CE by 0.00864, but its one-sided 95% validation
  delta upper bound was +0.00317 versus the +0.002 non-inferiority margin.
  The pre-registered quality decision is false despite the favorable mean.
- Job 1595625: native grouped experts plus Liger at microbatch 32 failed at
  candidate step 31 on a lower-memory RTX 4090: 701.7 MiB was reserved but
  unallocated, only 76.56 MiB was free, and a further 100 MiB was requested.
  This is a fragmentation failure, not evidence that the earlier high-memory
  mb32 frontier was fabricated.  Job 1595634 was cancelled while pending
  without running; the exact-node job 1595633 retry instead enables PyTorch's
  expandable CUDA segments and completed all six paired runs.
- Job 1595633: the exact-node native+Liger retry delivered a 34.90%--35.32%
  throughput gain in all three seed pairs and improved mean validation CE by
  0.00960.  Nevertheless, the mixed seed deltas gave a one-sided 95% upper
  bound of +0.01571 versus the +0.002 margin.  The direct quality gate is
  false, so native+Liger is not the default despite its systems gain.
- Non-batched CVCR remained above the gate even after credit fusion: job
  1595203 reached about 355k tok/s versus about 388k for Top-2.  Temporal
  batching was required; its added estimator variance is measured explicitly
  rather than hidden.
- The rank-8 predictor reduced held-out inactive-credit MSE by only 16.63%
  versus the default baseline, below the pre-registered 30% gate.  A rank-16
  predictor-only ablation reached 16.42%, so doubling factor rank did not close
  the gap.
- Although rank-8 CVCR reduced estimator variance 53.35% versus raw probing,
  it improved only 10.47% over the cheaper default control variate.  Variance
  reduction alone did not produce better language modelling.
- Exact equal-compute reruns found that local first-order credit was a poor
  downstream proxy: Pearson correlation 0.206 and Spearman correlation 0.199.
- A pure Top-2 comparison gave CVCR +0.007324 nats/token worse CE, but a
  predictor-only negative control showed that changing the DDP/optimizer graph
  can create a much larger numerical training-trajectory divergence even when
  credit injection is zero.  That comparison is retained as a product-baseline
  result, not claimed as the causal effect of CVCR.
- The strict contrast used identical rank-8 probes, predictor, DDP graph,
  initialization seed, data order and schedule, changing only the credit
  coefficient from 0 to 0.05.  CVCR was worse by +0.023548 nats/token on 256
  frozen packed blocks; paired 95% bootstrap interval
  [+0.021332, +0.025851].
- Because predictor usefulness and strict quality gates both failed, the
  7.002B/1.189B-active design was not submitted for training.  Scaling this
  candidate would violate the research programme's own stop rule.
