# CVCR oracle-gate decision

Date: 2026-09-16  
Hardware: 4 x RTX 4090 for training; 1 x RTX 4090 for oracle/paired evaluation  
Frozen data revision: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`  
Proxy: 277,801,728 deployed parameters, 79,620,864 active parameters/token  
Training budget: 199,753,728 prediction tokens per method

## Decision

**Stop before 7B and redesign the estimator.**  Systems efficiency passes, but
the predictor-usefulness and strict quality gates fail.  `moe_7b_cvcr.json`
remains a parameter-count-checked design configuration, not a trained model.

## Gate table

| Gate | Pre-registered criterion | Observation | Result |
|---|---|---|---|
| Counterfactual gap exists | measurable inactive-route opportunity | 40.63% of exact alternatives positive; 53.13% of sampled tokens had a positive alternative | yes, small gap |
| Predictor usefulness | at least 30% inactive-credit MSE reduction vs default | rank-8: 16.63%; rank-16: 16.42% | fail |
| Estimator variance | at least 20% reduction vs raw probes at equal cost | rank-8: 53.35% | pass |
| Local proxy fidelity | local credit should predict downstream reruns | Pearson 0.206; Spearman 0.199 | fail/weak |
| Proxy quality | positive paired CE improvement | +0.023548 nats/token; 95% CI [+0.021332, +0.025851] vs strict control | fail |
| Systems | no more than 5% wall overhead | 4.48% vs pure Top-2 over 361 post-warm-up steps | pass |
| Inference | ordinary Top-2, no predictor/probe execution | inference state strips all training-only tensors; eval-path equality covered by tests | pass structurally |

## Strict comparison

The causal quality baseline is the rank-8 instrumented control, not pure
Top-2.  Both control and CVCR use the same predictor, inactive probes,
temporal interval, DDP parameter graph, initialization seed, data order,
optimizer and schedule.  The only intended treatment difference is
`cvcr_coefficient` 0 versus 0.05.

| Run | Slurm job | Checkpoint SHA-256 | Frozen-block CE |
|---|---:|---|---:|
| Rank-8 control | 1595300 | `6832a5e9f7ba5e8c06fb6f167f0e17c8d788ca5d1168eff01cbf385fb26e2d41` | 4.5156369 |
| Rank-8 CVCR | 1595224 | `f399001d8e71fb019d3d2e06e7a59c2c4d28ed29720921d41bf92a1b17782864` | 4.5391846 |

Paired evaluation job 1595316 used 256 frozen packed 2,049-token blocks and
10,000 bootstrap repetitions.  CVCR was better on 9.375% of blocks.  The
sampling unit is a packed block, not an IID token or necessarily an original
source document.

## Systems accounting

Full-run medians after 20 warm-up steps:

| Method | tok/s | Active MFU | Hardware-work MFU | Overhead vs Top-2 |
|---|---:|---:|---:|---:|
| Pure Top-2, job 1595223 | 379,433.93 | 28.464% | 28.464% | baseline |
| Rank-8 control, job 1595300 | 370,543.24 | 27.797% | 27.875% | 2.399% |
| Rank-8 CVCR, job 1595224 | 363,177.24 | 27.244% | 27.321% | 4.476% |

The added credit computation costs 2.028% relative to the instrumented
control.  Short performance screens reached 387,867.89 tok/s for Top-2 and
378,696.31 tok/s for CVCR, but full-run medians are used for the gate.

## Mechanism diagnostics

Rank-8 oracle job 1595249 evaluated every expert on 4 batches x 8 layers x 256
tokens.  Predicted-credit MSE was `1.9322565e-08` versus
`2.31776848e-08` for the default.  Predicted-control variance was 46.65% of raw
HT variance, but 89.53% of the default-control variance.

Exact reroute job 1595251 performed 64 full downstream reruns, replacing the
weaker Top-2 expert with router rank 3 or 4.  Mean positive equal-compute
regret was `5.61774e-05` nats/token.  The weak correlation between local and
actual regret explains why a lower-variance local estimator did not translate
to better CE.

## Claim boundary

This is a one-seed, oracle-scale negative result.  It proves neither that every
CVCR variant fails nor that a 7B MoE cannot work.  It does show that this
specified low-rank local-credit mechanism has not earned the requested 7B
compute.  The programme calls for raw probing or a different router estimator
when the predictor gate fails.
