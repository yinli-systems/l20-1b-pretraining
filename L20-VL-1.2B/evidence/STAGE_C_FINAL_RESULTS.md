# Stage C Final Results — Frozen Test Evaluation

Status: complete
Evaluation date: 2026-09-13
Hardware: one NVIDIA L20 (46,068 MiB)
Primary test unit: 3,000 held-out scene families, paired by family
Uncertainty: 10,000-replicate paired cluster bootstrap, 95% CI

## Outcome

The strongest observed result is the spatially anchored 49-token model trained
with answer supervision alone. It compresses 196 visual tokens to 49 (4.0x),
scores 75.733% on the frozen controlled test, and exceeds the selected 196-token
teacher by +2.400 percentage points, 95% CI [1.067, 3.733]. This passes both the
predeclared statistical rule (lower CI > 0) and practical rule (estimate >= 2 pp).

This comparison does **not yet isolate compression as the cause**: the 49-token
student received an additional 875-step continuation from the selected teacher,
while the reported 196-token teacher did not receive a matched continuation.
Accordingly, the numerical result is valid but the causal description
"compression improved accuracy" is withheld until matched `native_196` and
capacity-matched `spatial_196` continuation controls are complete.

The proposed counterfactual evidence-delta objective is not supported. It scores
73.933%, versus 74.433% for its matched strong baseline: -0.500 pp, 95% CI
[-1.367, 0.400]. The shuffled-pair control was therefore not triggered by the
development gate.

## Frozen test ranking

| Rank | Arm | Visual tokens | True image | Random image | No image | True - random (95% CI) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | answer_49 | 49 | 75.733% | 13.400% | 0.000% | +62.333 pp [60.400, 64.267] |
| 2 | kd_49 | 49 | 74.567% | 13.367% | 0.000% | +61.200 pp [59.233, 63.133] |
| 3 | strong_49 | 49 | 74.433% | 13.367% | 0.000% | +61.067 pp [59.100, 62.967] |
| 4 | proposed_49 | 49 | 73.933% | 13.200% | 0.000% | +60.733 pp [58.733, 62.700] |
| 5 | full_196 teacher | 196 | 73.333% | 13.700% | 0.000% | +59.633 pp [57.667, 61.667] |

The random-image and no-image controls show that the result is not attainable
from question text alone in this controlled protocol.

## Confirmatory paired comparisons

| Candidate - baseline | Estimate | 95% CI | Statistically positive | >= 2 pp |
|---|---:|---:|:---:|:---:|
| answer_49 - full_196 | +2.400 pp | [1.067, 3.733] | yes | yes |
| answer_49 - kd_49 | +1.167 pp | [0.167, 2.200] | yes | no |
| answer_49 - strong_49 | +1.300 pp | [0.233, 2.400] | yes | no |
| answer_49 - proposed_49 | +1.800 pp | [0.767, 2.833] | yes | no |
| proposed_49 - strong_49 | -0.500 pp | [-1.367, 0.400] | no | no |

## answer_49 task breakdown versus full_196

| Task | answer_49 | full_196 | Paired difference (95% CI) |
|---|---:|---:|---:|
| Above/below | 100.000% | 100.000% | 0.000 pp [0.000, 0.000] |
| Color binding | 35.500% | 33.500% | +2.000 pp [-2.667, 6.833] |
| Count | 98.000% | 95.333% | +2.667 pp [1.000, 4.333] |
| Left/right | 99.667% | 95.167% | +4.500 pp [2.833, 6.333] |
| Shape binding | 45.500% | 42.667% | +2.833 pp [-1.000, 7.000] |

## Efficiency

All four Stage C arms used the same 14,000 training families, 42,000 image
examples, 875 optimizer steps, data order, seed, and selection schedule.

| Arm | Wall time | Input+visual tok/s | Peak allocated VRAM |
|---|---:|---:|---:|
| answer_49 | 801.949 s | 6,976.01 | 4.337 GiB |
| kd_49 | 801.830 s | 6,977.05 | 4.337 GiB |
| strong_49 | 801.133 s | 6,983.12 | 4.337 GiB |
| proposed_49 | 801.683 s | 6,978.33 | 4.337 GiB |

The frozen inference benchmark used an ABBA order, separate warmups, real test
images, eight families / 16 candidate sequences per batch, and 60 measured
repetitions per arm.

| GPU path | full_196 median | answer_49 median | Reduction | Speedup |
|---|---:|---:|---:|---:|
| Image tensor -> vision -> bridge -> LM logits | 149.135 ms | 53.679 ms | 64.006% | 2.778x |
| Cached vision -> bridge -> LM logits | 141.437 ms | 45.403 ms | 67.899% | 3.115x |

The latency result excludes disk I/O, CPU image decode/preprocessing,
autoregressive decoding, concurrency, and serving overhead.

## Integrity anchors

- Data manifest: `6f21e01d31d8fc3035ca22219d0484a8fb029a30b2cddec34ab726585e5dfede`
- Matched-matrix protocol: `6c619268257ce6a018abbeafae66573978cfad452ba0c8c8fd1da514726ac70c`
- Test-unseal protocol: `43c01d0e4d037bca9d1d9b4862940a98d5f1b95a51468f69e0f02619d65d7587`
- Teacher cache: `9f59cf8869b68adee8c7c87393d948c4f6bea21b2f6cd8d3ae9ce2a1cccf2ae3`
- full_196 test receipt: `0e1f2b3af9e341294f56db9b22c9624028db0240e2421b6a42282849176121ad`
- answer_49 test receipt: `69ef503bead1792981350d82b4244ab791462797c12f9f220237c66dbfd3c47a`
- kd_49 test receipt: `4947cc0883012c135d2bbdb5de516981e0af28589b1d86ccdf56a1d516b20ac0`
- strong_49 test receipt: `ce2906f070ca204a9b1ac8844c9d3f61ae974c36a4aff5a971ab941b338858b7`
- proposed_49 test receipt: `87a24fd954bb9feb47a2e2929774de803c0bc754dce20074ea0b37d01f21c474`
- answer_49 versus full_196 receipt: `309eb69ed2d2785c9dde6117f1ff987ba038ea6464e6702013b81afeb82fb001`
- proposed_49 versus strong_49 receipt: `39ce79ebf43b3848bdb46b831ca3b215132f6cb1d34796abb3e5c2ff9fd781a3`
- Latency receipt: `5bec1c28a3d9f11e85b27b36374d3f61fc7f5b8b7ca4589057b9991ec3d6d6ac`

Selected bridge checkpoints:

- full_196: `22cdd610fd52455ff43d378456fc097f0212f873ab2f1845cd4fd8f1198320bd`
- answer_49: `69de178d68f20ac3899a85e1581e837caa0d775e655d66853644b610b29bb7a1`
- kd_49: `11f140444385aaf50f80b2a04a85e76ff00a2cee0ca520d62e79a2e1fe722df7`
- strong_49: `316f78b85234f068ea2e484151c4a4fd0b82af96586d434dc1fa4a48620de1e0`
- proposed_49: `71921ef87715d72c8d2cac728e20497dc8bd278d1610f49e9f5aea4e71230c04`

## Claim boundary

This is a one-seed, controlled synthetic-domain experiment. It is strong
evidence for visual dependence and an accuracy/latency association at 49 tokens,
but it does not yet establish a compression-caused accuracy gain because the
matched continued-196 controls were absent. It is also not evidence of real-image
transfer, public VLM benchmark quality, broad instruction-following, multi-seed
robustness, or production serving performance. The language and vision parents
were pretrained; this Stage C model is post-training on top of those frozen
parents, not a VLM trained from scratch. The failed proposed objective is
retained as a negative result.
