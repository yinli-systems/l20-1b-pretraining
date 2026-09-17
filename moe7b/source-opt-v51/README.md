# CVCR MoE research implementation

This directory implements and evaluates the first two gates of the attached 7B
MoE research programme: a dropless Top-2 proxy and Control-Variate
Counterfactual Routing (CVCR), with exact inclusion probabilities,
training-only probes/predictors, paired initialization, frozen packed-data
verification, throughput/MFU receipts, and an exhaustive 16-expert
local-credit oracle.

It is deliberately not described as a finished 7B model.  A performance
screen proves runtime only; a 0.2B-token warm-up plus exhaustive oracle proves
only the mechanism gate.  The programme permits a 7B pilot only after the
predictor, estimator-variance, routing-regret, quality, and systems gates pass.

## Implemented invariants

- Ordinary deterministic dropless Top-2 is the deployed forward path.
- Primary CVCR uses an input-conditioned rank-8 control variate and real
  inactive-expert probes.  A rank-16 predictor-only ablation is included.
  Probe expert weights are detached, so probes add no expert parameter
  gradient.
- Uniform probes use a shared expert and fixed compute with replacement.  The
  Horvitz-Thompson inclusion probability accounts for the number of eligible
  experts, per-expert inactive-token count, and duplicate draws.
- Temporally batched CVCR executes once every four microbatches with four times
  the conditional probe budget and auxiliary coefficient.  It preserves the
  average probe FLOPs and expected auxiliary update while amortizing fragmented
  launches.  The oracle reports both conditional and temporal estimator
  variance because temporal batching increases gradient variance.
- BF16 expert/attention compute, FP32 router softmax and reductions, Flash
  SDPA, and separately compiled credit backward are used on CUDA. Native
  jagged `torch.nn.functional.grouped_mm` remains the reference backend;
  `scattermoe` is an opt-in performance candidate and never activates by
  silent fallback.
- `inference_state_dict()` removes every predictor and sampler tensor.
- The exact 7B design configuration computes 7,002,228,736 deployed parameters
  and 1,188,923,392 active parameters per token.

## Verified systems gate

On four RTX 4090 GPUs, sequence length 2,048, microbatch 8/GPU, accumulation 8:

| Method | Median tok/s | Legacy reported MFU | Wall overhead |
|---|---:|---:|---:|
| Top-2 | 379,433.93 | 28.46% | baseline |
| Rank-8 instrumented control (`lambda=0`) | 370,543.24 | 27.80% | 2.40% |
| Rank-8 CVCR interval-4 (`lambda=0.05`) | 363,177.24 | 27.24% | 4.48% |

These medians use 361 steps after a 20-step warm-up from the complete 199.75M
token jobs 1595223, 1595300 and 1595224.  The fastest short screens reached
387,867.89 tok/s for Top-2 and 378,696.31 tok/s for CVCR.  CVCR therefore
passes the programme's <=5% systems gate.

The archived 28.46% value used 209.5 TFLOP/s/GPU and a full-square attention
numerator. It is retained only as `legacy_reported_mfu`. For the unchanged
Top-2 run, NVIDIA's 165.2 dense-BF16 TFLOP/s RTX 4090 rating gives 36.10% with
the legacy numerator, while exact causal-pair useful matmuls give 31.76%.
These are accounting corrections, not speedups. New runs emit all three
versioned metrics.

## Consumer-GPU backend audit

PyTorch 2.11's native BF16 grouped-GEMM fast-path predicate does not cover
SM89 (RTX 4090) or SM120 (RTX 5090); the ragged fallback copies offsets to the
CPU and launches per-group GEMMs. The first optimization gate therefore tests
a pinned ScatterMoE Triton backend with complete forward, input-gradient, and
weight-gradient validation. It also removes the redundant second sort used to
invert the assignment permutation.

Install the exact optional dependency on a networked login node:

```bash
python -m pip install -r requirements-performance.txt
```

Then run the falsification benchmark inside a real GPU allocation:

```bash
python scripts/benchmark_expert_backends.py \
  --config configs/model/oracle_top2.json \
  --tokens 16384 32768 \
  --distributions balanced skewed empty \
  --warmup 10 --steps 50 --trace-dir traces/expert-backends
```

The candidate is rejected unless forward, dX, both dW tensors, empty/skewed
experts, and one optimizer update are acceptable and the representative expert
block is at least 1.3x faster. A full-step improvement requires a separate
same-node ABBA screen; backend microbenchmarks alone do not establish a new
training record.

## Performance optimization outcome (2026-09-17)

The optimized proxy path combines the pinned ScatterMoE backend, Liger 0.8.2
linear cross-entropy, and microbatch 32 / accumulation 2.  It preserves the
global batch and exact ordered 64-record set per rank/update.  Its fastest
short frontier screen measured 569,734 tok/s and 47.69% causal-useful MFU.
Across the three 381-step sustained candidate runs it delivered
547,803--553,076 weighted tok/s, 548,968--554,569 median tok/s, and
45.96%--46.43% median causal-useful MFU.  The minimum paired throughput gain
over Scatter-eager was 28.62%.

That is the highest measured performance candidate, not the default.  Its
one-sided 95% validation-CE delta upper bound was +0.00317 versus the +0.002
non-inferiority margin.  The earlier native-to-Scatter E5 link also failed.

The arithmetic-preserving native-expert + Liger mb32 path delivered
480,666--480,704 weighted tok/s and 40.28%--40.29% median causal-useful MFU on
the lower-memory confirmation node, a minimum 34.90% paired speedup.  It too
failed the direct quality gate: mean validation CE improved by 0.00960, but
the one-sided 95% upper bound was +0.01571.  Therefore the promoted default
remains native grouped experts with eager cross-entropy at mb8/acc8.  These
proxy systems results do not reverse the failed CVCR science gates or
authorize a 7B run.  Full receipts and rejected variants are in
`reports/optimization_execution_20260917.md` and
`reports/negative_results.md`.

## Oracle and quality decision

The mechanism did not pass the pre-registered scientific gates:

- The rank-8 predictor reduced inactive-credit MSE by 16.63% versus the
  default baseline; rank-16 reduced it by 16.42%.  Both miss the required 30%.
- Rank-8 CVCR reduced estimator variance by 53.35% versus raw probing, but only
  by 10.47% versus the default control variate.
- Exact equal-compute reroutes found a real but small route gap, while the
  local first-order score correlated weakly with exact downstream reruns
  (Pearson 0.206, Spearman 0.199).
- Against the strict rank-8 instrumented control, CVCR worsened frozen-block
  CE by +0.023548 nats/token; paired 95% bootstrap interval
  [+0.021332, +0.025851].

The 7B configuration is therefore **not promoted to training**.  The complete
gate record is in `reports/oracle_gate.md`; failures remain in
`reports/negative_results.md`.

## Local verification

```bash
python3 -m pytest -q tests
```

The ParaCloud scripts copy source to node-local temporary storage, put compile
caches there, use no Slurm memory flags on the 4090 partition, and remove only
their exact temporary directory on exit.

## Key entry points

- `cvcr_moe/model.py`: GQA/RoPE decoder, grouped experts, Top-2 and CVCR paths.
- `cvcr_moe/cvcr.py`: predictor, uniform/ABP sampling, HT correction, compiled
  auxiliary router credit.
- `cvcr_moe/train.py`: paired DDP pretraining, receipts, validation, resume and
  versioned MFU accounting.
- `scripts/benchmark_expert_backends.py`: balanced, skewed and empty-expert
  forward/dX/dW comparison of the native reference and pinned ScatterMoE.
- `scripts/oracle_diagnostics.py`: exhaustive local-credit MSE, rank, recall,
  regret and conditional/temporal variance diagnostics.
- `slurm/oracle_warmup_4x4090.sbatch`: paired 0.2B-token mechanism warm-up.
- `slurm/top2_scattermoe_liger_fullpass_4x4090.sbatch`: resumable one-pass
  8.3765B-token Top-2 proxy training with the fastest measured systems path;
  it verifies checkpoint hashes before every resume.
- `configs/model/moe_7b_cvcr.json`: exact requested 7B/1.189B-active shape; it is
  a configuration and parameter-count assertion, not a trained checkpoint.

## Claim boundary

The code's local-credit estimator is conditionally unbiased under its recorded
sampling design.  The injected auxiliary router gradient is not claimed to be
an unbiased gradient of discrete Top-2 language-model routing.  Local
first-order oracle regret is not exact downstream counterfactual loss.
The CE confidence interval samples frozen packed blocks, not IID tokens or
original source documents.  Only one seed was run because the earlier
predictor and quality gates failed; no multi-seed proxy or 7B claim is made.
