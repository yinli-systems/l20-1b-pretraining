# ParaCloud throughput/MFU optimization execution — 2026-09-17

## Claim boundary

This report covers the 8-layer proxy on four RTX 4090 GPUs.  It does not
promote CVCR, authorize the 7B run, or establish 5090 feasibility.  The CVCR
science gates remain failed and no 7B pretraining was launched.

MFU is reported with three explicit conventions.  `legacy_reported_mfu`
retains the archived 209.5 TFLOP/s/GPU denominator and full-square attention
numerator.  `rated_full_square_mfu` uses NVIDIA's 165.2 dense-BF16
TFLOP/s/GPU rating for RTX 4090.  `causal_useful_matmul_mfu` additionally uses
exact causal-pair attention matmuls; it is the primary useful-work metric here.

## Implemented performance path

- Replaced the consumer-GPU `grouped_mm` fallback candidate with a pinned,
  device-resident ScatterMoE backend at commit
  `47b5e1502e5a10e82c8e5945d761b877849871e7`; the native path remains the
  correctness reference and default unless a quality gate passes.
- Removed the second assignment sort by constructing the inverse permutation
  with a device scatter.
- Added an opt-in Liger 0.8.2 fused linear cross-entropy path.  Its wheel is
  pinned by SHA-256
  `84c0a7bc9bf4d4cf8ea5ba89ff84d28686afc94215b220851d9f57dc87852741`.
  A deliberate Dynamo graph boundary works around the torch 2.11 fake-tensor
  incompatibility while retaining the fused CUDA operator.
- Used the loss-memory saving to raise the per-rank training microbatch from 8
  to 32 while reducing accumulation from 8 to 2.  The global batch, schedule,
  and ordered set of 64 records/rank/update are unchanged.  Validation is
  independently fixed at microbatch 8 so candidate and baseline evaluate the
  same number of frozen records.
- Added versioned MFU fields, optional backend validation, one-update parity,
  same-allocation ABBA screens, microbatch-frontier screens, and three-seed
  sustained confirmation.

## Correctness and systems gates

| Gate | Receipt | Result |
|---|---|---|
| Local/remote tests | source v18, plus final local suite | 18/18 local and 18/18 on the final ParaCloud source snapshot |
| Expert operator | job 1595477 | ScatterMoE exact reported output/dX/dW errors across balanced, skewed, and empty-expert cases; 1.23x–1.65x expert-block speedup |
| Liger operator | job 1595527 | loss abs error 9.54e-7; hidden-grad max abs 1.79e-7; weight-grad max abs 3.27e-6; 5.50 GB operator-memory reduction |
| Scatter one-update | jobs 1595505/1595512 | finite; Scatter max gradient relative-L2 0.0833 versus native-null 0.00639; requires outcome-level quality gate |
| Scatter+Liger mb32 compiled one-update | job 1595592 | gradient-norm delta +0.00256; max gradient relative-L2 0.0687; same-record post-update CE delta +0.00507 |
| Native+Liger mb32 compiled one-update | job 1595601 | gradient-norm delta +0.00102; max gradient relative-L2 0.0665; same-record post-update CE delta +0.00127 |

## Measured optimization ladder

| Path | Evidence window | Throughput | Causal useful MFU | Decision |
|---|---|---:|---:|---|
| Native eager mb8/acc8 | job 1595521, three sustained baselines | 376.5k–378.5k weighted tok/s | about 31.6% median | reference |
| Scatter eager mb8/acc8 | job 1595521, three sustained candidates | 423.7k–424.8k weighted tok/s | about 35.6% median | systems win, quality not promoted |
| Scatter+Liger mb16/acc4 | job 1595573 ABBA | 517,141 weighted tok/s | 43.36% median | superseded by mb32 |
| Scatter+Liger mb32/acc2 | job 1595607, three sustained candidates | 547,803–553,076 weighted tok/s | 45.96%–46.43% median | fastest confirmed candidate; quality gate failed |
| Native+Liger mb32/acc2 | job 1595633, three sustained candidates | 480,666–480,704 weighted tok/s | 40.28%–40.29% median | direct quality gate failed |

The mb32 Scatter+Liger ABBA gain was 29.11% over its Scatter-eager baseline;
the paired-block bootstrap 95% ratio interval was [1.2840, 1.2950].  The
native+Liger ABBA gain was 38.32% over native eager, with ratio interval
[1.3757, 1.3883].  Absolute node-local short screens reached 569,734 tok/s and
47.69% causal-useful MFU for Scatter+Liger mb32, but that value is not used as
the confirmed record.

## Memory frontier and rejected tuning

- Scatter+Liger mb32 completed at 21.51 GB peak allocated and 23.00 GB peak
  reserved per rank.  Native+Liger mb32 completed at 22.32 GB allocated and
  23.20 GB reserved.
- On a lower-memory 22.03 GiB card, the first native+Liger mb32 E5 attempt
  fragmented at step 31 even though 701.7 MiB was reserved but unallocated.
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` fixed that exact-node
  failure: all three candidate runs in job 1595633 completed at 22.30 GB peak
  allocated and 22.70 GB peak reserved.
- mb64/acc1 failed on all four ranks: each had about 377 MiB free when a
  further 384 MiB allocation was required.  mb32/acc2 is the measured maximum
  arithmetic-preserving microbatch.
- DDP-before-compile gained only 0.12% (job 1595547) and was rejected.
- DDP bucket caps 8/25/100 MiB measured 433,371/435,188/425,789 tok/s (job
  1595562); the 25 MiB default remains.
- Whole-model `reduce-overhead` had already failed in jobs 1595206/1595207.
- Liger mb8 was only 1.6% faster than eager mb8.  Its value is memory headroom
  enabling mb32, not the fused head in isolation.

## Sustained quality decisions

The plain ScatterMoE E5 confirmation (job 1595521) reproduced a 12.24%–12.53%
throughput gain across all three seed pairs, but failed quality
non-inferiority: mean validation CE delta +0.01549 and one-sided 95% upper
bound +0.14655 versus the +0.002 margin.  ScatterMoE is therefore not promoted
as the default backend.

- Scatter+Liger mb32 E5, job 1595607: all three candidate runs delivered
  547,803--553,076 weighted tok/s and 45.96%--46.43% median causal-useful
  MFU.  The mean throughput ratio was 1.2983 and the minimum per-seed ratio
  was 1.2862.  Mean validation CE delta was -0.00864, but the one-sided 95%
  upper bound was +0.00317, narrowly above the +0.002 non-inferiority margin.
  The pre-registered promotion result is therefore false.
- Native+Liger mb32 E5, job 1595633: all three candidate runs delivered
  480,666--480,704 weighted tok/s and 40.28%--40.29% median causal-useful MFU.
  The mean throughput ratio was 1.3518 and the minimum per-seed ratio was
  1.3490.  Mean validation CE delta was -0.00960, but seed deltas
  (-0.02295, +0.00665, -0.01248) produced a one-sided 95% upper bound of
  +0.01571.  It therefore also fails the +0.002 non-inferiority gate and is
  not promoted.

The Scatter+Liger within-backend comparison did not pass; independently, the
rejected native-to-Scatter quality link would have prevented it from becoming
the default end-to-end path.  Native+Liger also failed its direct paired E5
quality gate.  The deployable default therefore remains native grouped
experts, eager cross-entropy, microbatch 8 / accumulation 8.

## Primary remote receipts

- `/ssd/scxi253/cvcr-research/backend-sustained/1595521/summary.json`
- `/ssd/scxi253/cvcr-research/liger-microbatch-frontier/1595576/`
- `/ssd/scxi253/cvcr-research/liger-mb32-abba/1595584/summary.json`
- `/ssd/scxi253/cvcr-research/native-liger-mb32-abba/1595602/summary.json`
- `/ssd/scxi253/cvcr-research/liger-mb32-sustained/1595607/`
- `/ssd/scxi253/cvcr-research/native-liger-mb32-sustained/1595633/`

Every completed remote experiment directory contains a `SHA256SUMS` receipt.
The verified summary SHA-256 values are
`5276cb64d41cfce19c7915a03bf80273118c298f22498dd9cee45710b6387e29`
for job 1595607 and
`a80f00575a07d3539767d7a311cfab5b1fab8dabce36d95b70594a11ff447601`
for job 1595633.  The final immutable source snapshot is
`/ssd/scxi253/cvcr-research/source-opt-20260917-v18`.
