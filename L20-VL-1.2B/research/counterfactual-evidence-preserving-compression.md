# Counterfactual Evidence-Preserving Visual Compression

**Research and execution decision — 2026-09-13**

## Executive decision

The project will test one narrow hypothesis: at an identical 49-token visual
budget, can a compressed student preserve the full-token model's change in
answer preference across a controlled, answer-changing image edit better than
ordinary distillation plus counterfactual supervision and invariance?

This is an unproven hypothesis, not a novelty claim. Small VLMs, token
compression, counterfactual training, sensitivity/invariance supervision,
full-token distillation, and relational distillation all have direct prior art.
The experiment is worth running only if it first isolates a genuine
compression-induced failure and then beats a strong combined baseline.

The implementation and controlled diagnostic corpus are ready. Formal training
is not authorized: the full-token VLM foundation still lacks an admitted
training mixture and has not passed the frozen visual floor.

## What prior work already covers

| Area | Closest evidence | Consequence for this project |
|---|---|---|
| Compact VLM construction | [SmolVLM](https://huggingface.co/blog/smolervlm) | A small VLM on modest hardware is engineering, not the contribution. |
| Parameter-free token compression | [DeCo](https://arxiv.org/abs/2405.20985) | Two-dimensional adaptive pooling must be a baseline. |
| Learned visual compression | [VoCo-LLaMA](https://arxiv.org/abs/2406.12275), [MetaCompress](https://arxiv.org/abs/2603.21701) | A learned projector alone is not new. |
| Visual-necessity data selection | [VisNec](https://arxiv.org/abs/2603.01195) | Image/no-image usefulness filtering is separate prior art. |
| Counterfactual visual learning | [CF-VLM](https://arxiv.org/abs/2506.17267), [VIGIL](https://arxiv.org/abs/2606.26387) | “The answer should change when the image changes” is not new by itself. |
| Sensitivity and invariance | [SIVA-RL](https://arxiv.org/abs/2607.13931) | Relevant-change sensitivity plus irrelevant-change invariance is a strong baseline component. |
| Full-token teacher distillation | [Token-Budget Distillation](https://arxiv.org/abs/2608.28138) | Ordinary answer-region KL and margin distillation must be controlled. |
| Relational distillation | [RKD](https://arxiv.org/abs/1904.05068) | Cross-example relations are not claimed as a new general idea. |
| Compression-related spatial loss | [3D spatial-understanding analysis](https://arxiv.org/abs/2605.20448) | Supports the motivation, but does not establish the failure on this model. |
| Text-to-vision transfer | [What Transfers from Text to Vision?](https://arxiv.org/abs/2608.00013) | The 1.1B text checkpoint is a controlled substrate, not automatic evidence of VLM quality. |

The defensible contribution boundary is therefore narrower: a controlled test
of whether explicitly matching *cross-image answer-preference deltas* allocates
a limited visual bottleneck more effectively than the strongest same-budget
combination of existing supervision signals.

## Frozen model and resolution choice

The language parent is `AliceYin/L20-1B-20B-Base` with 1,100,048,384
parameters. It and its tokenizer were pretrained from random initialization.
The vision parent is the separately pretrained
`google/siglip2-base-patch16-224`, pinned to revision
`75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` and vision-weight SHA-256
`612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b`.
Consequently, the combined VLM must not be described as fully pretrained from
scratch.

Although an official [SigLIP2 256-pixel checkpoint](https://huggingface.co/google/siglip2-base-patch16-256)
exists, version 1 retains the already verified 224-pixel encoder. This gives a
clean 14 by 14 grid: 196 full tokens and exactly 49 tokens after 2 by 2 spatial
pooling. Switching backbone weights, input resolution, and compression at once
would confound the mechanism test. A 256-to-64 replication is deferred until a
full-token 224 model is shown to be resolution-limited.

## Failure definition

For a base image and an answer-changing edit, define pair correctness as

```text
pair_correct = base_correct AND edited_correct
```

An error is compression-induced only when the full 196-token model is correct
on both images and the compressed 49-token model is wrong on at least one. A
full-token error is evidence of a more basic encoder, alignment, data, or
language problem; it is not evidence against the compressor.

Before compression experiments, the full-token model must reach at least 70%
joint accuracy on the paired synthetic diagnostic, and both true-vs-no-image
and true-vs-random-image gains must have lower 95% confidence bounds above
zero. If the full-versus-compressed joint-accuracy gap is below two percentage
points, this project stops because there is too little bottleneck failure to
repair.

## Candidate objective

For image `I`, question `q`, correct answer `y`, and alternative `y_alt`, define
answer preference

```text
r(I) = log p(y | I, q) - log p(y_alt | I, q)
```

For an answer-changing edit `I'`, define the cross-image preference difference

```text
d = r(I) - r(I')
```

The proposed additional term is

```text
L_delta = Huber(d_student - stop_gradient(d_full_token))
```

It is applied only when the full-token reference answers both images correctly.
Ground-truth answer supervision remains available on other samples, but an
incorrect teacher preference is never distilled. Teacher-pair coverage is a
mandatory reported metric.

This term cannot add information that perfect per-image distribution matching
already contains. The empirical hypothesis is only that, under finite
capacity and compute, the explicit pairwise target prioritizes decision-relevant
differences. The shuffled-pair control directly tests that explanation.

## Frozen experiment matrix

All primary 49-token runs use the same examples, prediction-token budget,
optimizer, schedule, backbone checkpoint, and visual-token count. Wall-clock
time is reported rather than artificially equalized.

1. Full 196-token reference.
2. Parameter-free 2D adaptive pooling to 49 tokens, in the spirit of DeCo.
3. Learned 49-token compressor with answer supervision.
4. Learned compressor plus ordinary full-token distillation.
5. Strong baseline: answer supervision, ordinary distillation,
   counterfactual supervision, and invariance.
6. Proposed: the strong baseline plus cross-image preference-delta matching.
7. Mandatory mechanism control: shuffle pair links while preserving all images,
   labels, and compute.
8. Mandatory scope control: add the delta objective to the full-token branch.

If shuffled pairs match the proposed gain, the mechanism is rejected. If the
full-token branch gains equally, the result is reframed as general visual
grounding rather than compression repair.

## Controlled diagnostic corpus

The deterministic generator produces 5,000 scene families and three images per
family: base, answer-changing edit, and answer-invariant background edit. It
covers count, color binding, shape binding, left/right, and above/below with
balanced yes/no answers. Splits are scene-family atomic and stratified by task
and base answer: 70% train, 15% development, and 15% test.

Each manifest row stores the complete scene state, edit description, independent
oracle query, render seeds, question, answers, split, and image hashes. This is
informed by controlled visual reasoning work such as
[CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) and intervention-focused
[EditCLEVR](https://github.com/torux-bughunter/EditCLEVR), but the generator is
not presented as a research contribution.

The final corpus contains 15,000 unique PNG hashes and 92,358,381 exact image
bytes. Every task has 700/150/150 train/development/test families and exactly
balanced answers. The manifest SHA-256 is
`df65dbef20fb41d8d08f76a14fb9d3bf5f778a82c2210e56287cb8c221508d7e`.

Two failed builds are preserved:

- Attempt 1 detected one global exact-image collision and was rejected before
  admission.
- Attempt 2 passed exact deduplication, but human review found target/distractor
  overlap in some relation edits. It was rejected and the generator was fixed
  with relation-specific safe distractor positions plus exhaustive unit checks
  over both answers, both relation tasks, all variants, and 100 seeds.

The accepted build passed a deterministic 60-family/180-image stratified human
contact-sheet review. This admits it only for controlled mechanism diagnosis
after Stage A. It does not prove exhaustive correctness or natural-image
generalization and does not authorize parameter updates.

## Metrics and statistics

The primary metric is base-plus-answer-changing-edit joint accuracy. The primary
comparison is proposed method minus the strong combination baseline.

The confirmatory rule is frozen before predictions are produced:

- three independent training seeds;
- paired cluster bootstrap by scene/original-image family;
- 10,000 resamples and a 95% interval;
- lower confidence bound above zero;
- point estimate at least +2.0 percentage points.

Secondary outcomes include compression-induced failure conditional on
full-token pair correctness, base-plus-invariant joint accuracy, ordinary
single-image accuracy, teacher coverage, per-task results, and true/no/random
image controls. A confidence interval crossing zero means “difference not
detected,” not equivalence; any equivalence claim needs a separately frozen
margin.

`evaluate_counterfactual_predictions.py` makes the paired unit explicit and
emits overall and per-task clustered intervals. It cannot convert unvalidated
predictions into a quality claim.

## Real-image transfer and efficiency

Programmatic scenes can establish a mechanism, not broad utility. Real-image
transfer candidates are [GQA](https://arxiv.org/abs/1902.09506) and
[MMVP](https://arxiv.org/abs/2401.06209), but neither may be used until exact
revisions, splits, licenses, exclusions, and artifact hashes are frozen. Test
images cannot enter training.

Efficiency claims require end-to-end time-to-first-token, language-prefill time,
peak VRAM, fixed-output latency, training GPU-hours, teacher-cache GPU-hours,
feature-cache GPU-hours, and data-filtering compute. A fourfold reduction in
visual-token count is not automatically a fourfold speedup. The earlier text
pretraining MFU and the existing forward-only bridge throughput are not VLM
training MFU.

## Single-L20 execution sequence

1. **Stage A — full-token foundation:** admit a lawful, high-quality multimodal
   training mixture; train conventional visual alignment and instruction
   adaptation; pass visual-use controls.
2. **Stage B — diagnosis:** score the 5,000 controlled pairs with full 196-token
   and matched 49-token branches; stop unless the frozen failure conditions hold.
3. **Stage C — four primary 49-token experiments:** answer supervision, ordinary
   KD, strong combination, and proposed delta objective; then run shuffled-pair
   and full-token controls.
4. **Stage D — external validity:** freeze real-image evaluations and replicate
   the decisive comparison on a second language backbone under a new budget.

Frozen vision features and full-token short-answer teacher scores may be cached
to reduce repeated compute, but their generation time and storage remain part
of the cost ledger. Freezing a backbone does not eliminate backward computation
through it when the bridge needs answer-loss gradients.

## Current evidence boundary

As of 2026-09-13, only architecture/loss implementation, unit validation,
forward-only systems tests, and diagnostic-data generation/audit have occurred.
No optimizer step has been authorized for this research method, and its quality,
MFU, latency benefit, real-image transfer, and novelty remain unproven. The next
legitimate action is Stage A data admission—not starting the proposed method on
an unqualified visual foundation.
