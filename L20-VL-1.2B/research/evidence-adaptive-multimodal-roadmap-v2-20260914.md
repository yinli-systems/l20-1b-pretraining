# Evidence-Adaptive 1B Multimodal Model

## Executive decision

The strongest direction is not a generic image chatbot. It is a compact visual
assistant with an explicit evidence contract:

1. inspect the complete image with 49 visual tokens;
2. point to the visual evidence or emit `STOP`;
3. acquire one high-resolution crop only when it is predicted to improve the
   answer;
4. return a short answer tied to that evidence, or abstain when the available
   evidence is insufficient.

This target connects the project’s strongest existing assets—its own 1.100B
language base, 196-to-49 spatial-query bridge, query-conditioned readout, and
counterfactual controls—to a useful product surface: photographs, screenshots,
documents, charts, and eventually short video. It also makes quality and cost
jointly measurable.

The literature review materially narrows the research claim. Dynamic
resolution, crop tools, direct visual-token pointing, uncertainty-aware token
budgets, and information-gap training all have close prior art. The project
should therefore claim no novelty for “zoom,” “point,” or “adaptive tokens.” A
defensible contribution would require evidence that a tiny model can predict
the *marginal value of one additional visual observation*, ground the answer to
the selected evidence, and improve a frozen quality-cost frontier under strong
same-data and same-compute controls.

## Current substrate and hard constraints

| Component | Current state | Consequence |
|---|---|---|
| Language parent | 1,100,048,384 parameters, pretrained from random initialization on about 20B text tokens | Retain it as the language identity; do not call the final VLM fully pretrained from scratch because the vision tower is separately pretrained. |
| Vision parent | SigLIP2-B/16 at 224 pixels, 196 patch tokens | Useful localization features exist, but tiny text can be destroyed before the bridge sees it. |
| Efficient bridge | Learned spatial-query compression from 196 to 49 visual tokens | Keep as the low-cost first view; do not force it to be the only view. |
| Query readout | A 659,521-parameter readout produced strong cross-fit development results on a small audited posture set | It supports question-conditioned addressing, not broad real-image capability. Independent confirmation failed at the data-quality gate. |
| Remote hardware | One NVIDIA L20, 46GB VRAM | Small adapters, frozen-feature caches, and bounded high-resolution pilots are realistic; full vision pretraining is not. |
| Remote storage | Approximately 8.7GB free at the latest live check | No large corpus is admitted or downloaded until source quality, size, and cleanup targets are frozen. |

The existing Stage-D result shows that 49 tokens can be efficient on the
controlled task, but weak OOD binding in both the 49- and 196-token branches
means the project cannot treat compression as the only bottleneck. The real
product model needs better data, language-side multimodal adaptation, and
evidence supervision in addition to a router.

## What the strongest nearby work already establishes

### High resolution and efficient vision

Qwen2-VL uses native dynamic resolution and multimodal rotary position
encoding; LLaVA-UHD uses image slices, compression, and a spatial schema; and
SigLIP2 provides native-aspect-ratio variants with improved localization and
dense features.^1,2,3 These results support variable-resolution inputs, but do
not show that a fixed 224-pixel model can recover text that was removed during
resizing.

FastVLM is especially relevant to the “extreme” endpoint objective. Its central
systems finding is that the vision encoder can dominate time-to-first-token at
high resolution, and that producing fewer, higher-quality visual tokens can be
more deployable than adding a complex pruning stack.^4 This means the project
must measure image preprocessing, vision encoding, language prefill, and decode
separately. A low LLM token count alone is not enough.

### Pointing and grounded answers

Molmo and PixMo demonstrate that dense human captions, Q&A, and pointing can
support an open multimodal model.^5 MolmoPoint goes further: instead of
generating coordinate text, a pointing token directly selects a visual token,
then refines within that region; it also includes a no-more-points class.^6 The
direct visual-token action and `STOP` design are therefore adopted as strong
engineering prior art, not presented as a new idea.

PointRL highlights a second issue: multiple coordinates may all be valid, and
multi-instance queries need coverage, count, and duplicate checks rather than
single-coordinate exact matching.^7 Bounding boxes or masks should remain
hidden verifier evidence, not be leaked into prompts.

Grounded Chain-of-Thought reports answer accuracy, grounding accuracy, and
answer-grounding consistency as separate quantities.^8 This is important here:
a plausible point is not proof that the answer used that evidence, and a correct
answer with a wrong point may be language bias rather than visual success.

### Active zoom and adaptive budget

ZoomEye and RegionFocus show that iterative region search can improve difficult
visual and GUI tasks without changing the base model.^9,10 AdaptVision trains a
coarse-to-fine crop policy with a separated tool and answer objective.^11
Learning to Focus and Precise Cropping uses an information-gap stage and a
grounding loss to make the model rely on the crop rather than merely invoking a
tool.^12 These are direct neighbors of any proposed active-zoom system.

Recent adaptive token work narrows the boundary further. PromPrune adapts the
saliency/coverage balance per sample; ET-Prune uses question-conditioned
evidence density and uncertainty to choose a token floor for text-rich inputs;
consequence-sensitive compression allocates visual budget according to the cost
of an error.^13,14,15 The project’s opportunity is not adaptive budgeting in
general. It is a small, calibrated, one-observation policy whose supervision is
the measured improvement caused by the extra evidence.

### Documents, screens, and video

UReader uses shape-adaptive crops and auxiliary text-reading/keypoint tasks for
document and screen understanding.^16 SmolDocling shows that a 256M model can be
competitive at a narrow, structured document-conversion task when its output
format and data are purpose-built.^17 ScreenSpot-Pro contains high-resolution,
small-target professional interfaces and is a demanding external grounding
benchmark rather than an easy UI toy set.^18

For video, SlowFast-LLaVA separates a low-frame-rate, spatially detailed stream
from a high-frame-rate, spatially compressed motion stream.^19 This is a better
fit for the project’s 2,048-token context than treating every video frame as a
full image. Video remains a later stage: first prove one-image evidence
acquisition and two-image change understanding.

### Faithfulness and multimodal post-training

mDPO identifies a failure mode in which response preference training can ignore
the image condition, and adds image-conditional preference supervision.^20 It is
relevant after a stable supervised model exists. It is not a reason to begin RL
or preference optimization before the answer, point, and crop actions are
reliably parseable.

## Recommended architecture

### Mode 0: global first view

Resize the complete image using the pinned SigLIP2 preprocessing and encode 196
patches. The existing spatial-query bridge emits 49 language-width tokens. This
preserves a cheap, globally aware first pass and a direct comparison with prior
experiments.

### Mode 1: evidence pointer or STOP

A small question-conditioned router scores the 196 pre-compression patch
features plus one `STOP` action. It does not generate coordinate strings. The
current rank-64 implementation adds 376,706 parameters and is independently
attachable; it does not mutate the frozen bridge or parent weights.

### Mode 2: one crop from the original image

If the router selects a patch and the calibrated gain threshold is passed, map
that patch to an expanded crop in the original image, resize the crop to the
vision tower’s native input, and encode it into another 49 tokens. The language
model then receives global context plus local evidence. The first experiment
permits only one crop so latency and causal attribution remain bounded.

### Mode 3: answer, evidence, or abstention

The output contract should be short and machine-checkable:

```text
<evidence patch=87>
<answer>42</answer>
```

or:

```text
<evidence stop>
<answer>blue</answer>
```

or:

```text
<abstain reason=insufficient_visual_evidence>
```

The internal pointer can later be rendered as a user-facing point or box. The
model should not print long visual reasoning by default; the evidence location
and a concise answer are easier to verify and cheaper for a 1B model.

## Counterfactual value-of-evidence supervision

For a training example with correct answer `y`, define:

```text
L_global = NLL(y | global_49, question)
L_crop_j = NLL(y | global_49, crop_j_49, question)
gain_j   = L_global - L_crop_j
```

The oracle crop comes from a train-only box, mask, point, or programmatic scene
record. Let `j*` be the best legal evidence crop. The target action is `j*` only
when `gain_j*` exceeds a threshold fixed on development data; otherwise the
target is `STOP`. The router jointly learns pointer cross-entropy and a robust
regression target for observed gain.

This is intentionally supervised and bounded. It avoids an expensive RL search
before the action space, responder, and measurements are trustworthy. It also
creates a falsifiable interpretation: if predicted gain does not rank useful
crops or does not calibrate out of distribution, the router has not learned
when to look closer.

The method still has adjacent prior art in information-gap crop training and
adaptive evidence allocation. A publication claim would require a more complete
novelty review and decisive empirical evidence; the current name describes the
mechanism, not established priority.

## Data program

### Tier A: deterministic mechanism data

Create high-resolution evidence-search scenes in which a small answer-bearing
field is linked to a globally visible object or panel. Each scene stores the
render program, target box, legal crops, answer, image hash, and atomic split.
Half the questions should require a crop; half should be answerable globally.
Counterfactuals should move the answer-bearing field while keeping the rest of
the image fixed, and should change the answer while preserving the target
address. This tier identifies whether the router can learn `POINT` versus
`STOP`; it cannot establish real-image utility.

### Tier B: licensed, verifiable training candidates

CoSyn-point has 68.1K synthetic pointing rows, embeds images in Parquet, and is
released under ODC-BY; its rendering code is also public.^21 CoSyn-400K provides
program-generated charts, documents, tables, diagrams, and QA, also under
ODC-BY.^22 These are better acquisition candidates than decentralized web URLs
because the image payloads and generation provenance are packaged. They still
require shard pinning, code/output inspection, exact deduplication, and a human
quality audit before admission.

RefCOCO/RefCOCO+/RefCOCOg provide human referring expressions over COCO objects
and are a strong bridge from synthetic pointing to natural images.^23 Their
manual acquisition and governing terms must be resolved before use. They should
be grouped by COCO image during splitting so multiple expressions for one image
do not inflate confidence.

### Tier C: external confirmation only

Use ScreenSpot-Pro for high-resolution GUI grounding and OCRBench-v2 for
text-rich breadth only after their exact artifacts, licenses, evaluator, and
contamination boundaries are pinned.^18,24 Use available development portions
for engineering and preserve an untouched test split for one final run. Do not
train on benchmark test images.

PixMo-Points-Eval was investigated as an external point benchmark. Its pinned
metadata contain 436 rows and 431 unique image hashes, with zero exact overlap
against 17 enumerated project manifests. However, a frozen 32-image,
model-blind availability preflight verified only 20 images; five requests failed
and seven payloads no longer matched the pinned hashes. The direct-URL path is
therefore rejected for scoring because an available-only subset would be
selection-biased. This is a dataset-delivery limitation, not a criticism of the
official annotations. See the [availability receipt](../evidence/pixmo-points-eval-image-availability-preflight-v1-negative.json).

## Training sequence on one L20

| Gate | Work | Maximum initial spend | Advance only if |
|---|---|---:|---|
| G0 | Source pinning, executable-label audit, split/decontamination checks, human sheets | CPU/network only | Every admitted source has a reproducible receipt and explicit governing terms. |
| G1 | Train an oracle-crop responder on a small synthetic high-resolution set | 0.5 GPU-hour | Crop input materially improves answer NLL and accuracy over global-only without answer leakage. |
| G2 | Freeze responder, cache global/oracle-crop outcomes, train rank-64 pointer/gain router | 0.5 GPU-hour | Point-or-STOP accuracy, gain ranking, and calibration beat entropy and attention baselines. |
| G3 | Compare always-stop, always-zoom, random crop, entropy, attention, pointer-only, and value-of-evidence policies | 1 GPU-hour | Candidate improves the accuracy-versus-visual-token frontier with paired uncertainty intervals. |
| G4 | Repeat the selected recipe on audited CoSyn-point/CoSyn-400K and one COCO-based set | 4 GPU-hours | Real and packaged synthetic results agree in direction; no benchmark test data enter selection. |
| G5 | Add bounded language LoRA and answer/evidence consistency training | 4 GPU-hours | Grounding and answer both improve; text retention and no/random-image controls pass. |
| G6 | High-resolution encoder arm: current crop pipeline versus SigLIP2 NaFlex, later FastViTHD if weights and deployment path are practical | profile first | Better end-to-end TTFT/accuracy frontier, not merely more pixels or a higher offline score. |
| G7 | Two-image comparison, then SlowFast short-video input | separate budget | Frame-order, frame-shuffle, and single-frame controls prove temporal use. |

These are caps for staged falsification, not a prediction that the result will
be competitive after a fixed number of hours. A stable 100-step profile is
required before any full-run ETA.

## Evaluation contract

### Quality

- answer exact match or task-specific score;
- point-in-box/mask and IoU where applicable;
- instance coverage and duplicate/missing point errors;
- answer-grounding consistency;
- true-image minus no-image and deranged-image controls;
- direct-answer versus oracle-crop upper bound;
- refusal precision, recall, and risk-coverage curve;
- expected calibration error and Brier score for answer confidence and predicted
  crop gain.

### Efficiency

- expected visual tokens per request;
- crops per request and crop acquisition rate;
- preprocessing, global vision, router, crop vision, LLM prefill, and fixed
  decode latency separately;
- time-to-first-token, peak VRAM, images per second, questions per second, and
  total GPU-hours;
- cold and warm runs on L20, then the actual target edge device.

### Statistics

Use paired bootstrap by original image or scene family, never by individual
question when multiple questions share an image. Report the point estimate and
95% interval for quality differences and for token/latency differences. For
multi-seed confirmation, expose both image-level and training-seed variation.
Do not call overlapping confidence intervals equivalence; predefine an
equivalence margin when that is the claim.

### Primary decision rule

The first confirmatory comparison should be value-of-evidence routing versus
the strongest non-value baseline under the same responder, source data, action
space, and total training GPU time. A candidate advances only if:

- the paired 95% confidence interval for answer accuracy improvement has a lower
  bound above zero;
- grounding accuracy does not regress beyond a frozen margin;
- expected visual tokens and measured TTFT are lower than always-zoom;
- predicted gain is calibrated enough to produce a monotonic risk-cost curve;
- no-image and mismatched-crop controls show that the answer depends on the
  acquired visual evidence.

## Product roadmap

1. **Grounded photo QA:** identify an object, point to it, answer briefly, and
   admit uncertainty.
2. **Screen assistant:** locate UI controls and read nearby text without taking
   actions. ScreenSpot-Pro is a grounding benchmark, not authorization for live
   computer control.
3. **Document and chart reader:** return structured fields, cells, equations,
   and source boxes. A narrow schema can create more value than unconstrained
   captioning at this model size.
4. **Visual comparison:** explain one verified change across two images, with an
   evidence point in each image.
5. **Short video:** use a SlowFast token allocation and evidence points across
   frames; keep clips bounded by the existing context length.
6. **Edge deployment:** distill or replace the vision encoder only after the
   reference model works; then quantize, export, and benchmark on the actual
   device.

Native audio, image generation, long video, and autonomous UI actions should
remain out of scope until these stages are reliable. Adding every modality at
once would consume the single-L20 budget while making failures impossible to
attribute.

## Immediate implementation status

The G1/G2 mechanism pilot is now complete. The deterministic v1 corpus contained
320 atomic scene families. Its frozen preprocessing floor was 47.92% row
accuracy from the global view versus 100% with the source-resolution oracle
crop, a +52.08 percentage-point difference with 95% CI [+35.42, +68.75]. An
isolated 10,501,122-parameter crop bridge then reached 100% local-evidence row
and family-joint accuracy on development and on its one-shot sealed
confirmation. Wrong-crop controls remained at 10.42% and 16.67%, respectively,
showing that the answer path used the selected evidence rather than merely a
second image input.

The first 376,706-parameter rank-64 router was rejected: POINT/STOP decisions
were perfect, but exact and within-one-patch localization were only 31.25% on
development. Error analysis showed 77.08% same-colour but only 41.67%
same-shape selections, consistent with an under-covered object-layout problem.
The gate was not weakened.

The unchanged router was therefore retrained on a coverage-scaled corpus with
2,048 train families, while 256 additional families were frozen before
training. All 4,608 rendered images had unique hashes, and a fixed 48-item
train-only renderer audit passed. The run updated only the router for 1,920
steps; the language model, SigLIP2 encoder, global bridge, crop bridge, and
language adapter were verified unchanged. Optimizer time was 19.13 seconds on
one NVIDIA L20, with 2.51 GiB peak allocated memory and 6,426.65 examples/s for
the cached router phase. These figures exclude feature caching and frozen 1.1B
teacher scoring.

All four predeclared scaled checkpoints passed the already-open v1 development
gate. The registered selection rule chose step 480: action, POINT recall, STOP,
exact-patch, and within-one-patch accuracy were all 100%; gain MAE was 0.529.
The predicted-crop policy answered all 48 local-evidence development rows
correctly. It answered only 29.17% of global STOP rows, exposing a separate
global-responder weakness.

The selected step-480 checkpoint then passed exactly one run on the untouched
256-family confirmation split. Across 512 rows, action, POINT recall, and STOP
accuracy were 100%; exact and within-one-patch localization were 96.09%; gain
MAE was 0.501. The actual predicted-crop answer path achieved 96.48% row and
95.31% family-joint accuracy on the 256 local-evidence rows, with candidate-order
invariance. Global STOP row accuracy was 37.11%, so total mixed-task row
accuracy was 66.80%. The confirmation receipt and raw predictions are retained
under `evidence/router-scale-v2/`.

The first audited natural-image transfer arm used 1,728 image-disjoint Open
Images families (1,536 train and 192 development), each paired as an unaltered
global STOP row and an official-box red-marked POINT row. A primary visual
audit verified all 48 fixed marked-image/crop cells. Because several broad
Open Images category names remained semantically ambiguous, this arm admitted
only localization and STOP supervision; category strings were never used as
answer-generation targets. The frozen 1.1B language model, SigLIP2 encoder,
language adapter, and synthetic parent router were hash checked before and
after training. Only 376,706 router parameters were updated. The 960-step
cached optimizer phase completed in 15.32 seconds on one NVIDIA L20 at 6,015.64
rows/s and 2.41 GiB peak allocated memory; feature caching and frozen-model
loading are excluded from those throughput figures.

The natural adaptation is a strong but formally negative development result.
Relative to the confirmed synthetic parent, step 720 increased official-box
hit rate from 36.46% to 80.73%, a paired +44.27 percentage points with a
deterministic image-level bootstrap 95% interval of [+34.90, +53.13]. It raised
within-one-center accuracy from 9.90% to 56.25%, +46.35 points with 95% interval
[+38.02, +54.69], while reaching 98.96% overall POINT/STOP action accuracy,
97.92% POINT recall, 100% STOP accuracy, and retaining 100% synthetic exact,
within-one, and STOP accuracy on the 96-row synthetic development set. However,
the frozen gate required at least 70% within-one-center accuracy. None of the
four checkpoints passed, so no checkpoint was selected and no natural
confirmation set was accessed. Performance plateaued from step 720 to step 960,
and errors were concentrated in large boxes without a directional bias. This
is consistent with the current independent per-patch scorer finding an object
region but not reliably aggregating both sides of a wide marked box to infer
its geometric center.

The next natural experiment should not game this diagnostic by detecting red
pixels with a hand-coded box parser. Instead it should test unmarked,
class-conditioned grounding on source images with a unique official instance,
using image-disjoint development and a separately acquired one-shot
confirmation set. A compact global spatial-refinement head or explicit
coordinate factorization is justified only if the unmarked baseline shows the
same center-aggregation failure. Answer generation, OCR, and screen/document
training remain separate gates.

This confirms the synthetic coarse-to-fine mechanism across a new seed; it does
not confirm natural-image transfer. G3 should now compare the frozen router to
always-stop, always-zoom, random-crop, entropy, and attention baselines under
measured cost. In parallel, the responder needs a separately split global-view
training arm. G4 remains the decisive next capability gate: audited natural or
packaged-synthetic grounding data, followed by COCO-based and text-rich/OCR
evaluation without using benchmark test data for selection.

## Sources

1. Wang et al. “[Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191).” 2024.
2. Guo et al. “[LLaVA-UHD: an LMM Perceiving Any Aspect Ratio and High-Resolution Images](https://arxiv.org/abs/2403.11703).” 2024.
3. Tschannen et al. “[SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786).” 2025.
4. Vasu et al. “[FastVLM: Efficient Vision Encoding for Vision Language Models](https://machinelearning.apple.com/research/fast-vision-language-models).” CVPR 2025.
5. Deitke et al. “[Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models](https://arxiv.org/abs/2409.17146).” 2024.
6. Clark et al. “[MolmoPoint: Better Pointing for VLMs with Grounding Tokens](https://arxiv.org/abs/2603.28069).” 2026.
7. Su et al. “[PointRL: Learning Point-Level Vision-Language Grounding from Verifiable Annotation Evidence](https://arxiv.org/abs/2608.25299).” 2026.
8. Wu et al. “[Grounded Chain-of-Thought for Multimodal Large Language Models](https://arxiv.org/abs/2503.12799).” 2025.
9. Du et al. “[ZoomEye: Enhancing Multimodal LLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration](https://arxiv.org/abs/2411.16044).” 2024.
10. Luo et al. “[Visual Test-time Scaling for GUI Agent Grounding](https://arxiv.org/abs/2505.00684).” 2025.
11. Lin et al. “[AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition](https://arxiv.org/abs/2512.03794).” 2025.
12. Zhao et al. “[Learning to Focus and Precise Cropping](https://arxiv.org/abs/2603.27494).” 2026.
13. Lee et al. “[Balancing Saliency and Coverage: Semantic Prominence-Aware Budgeting for Visual Token Compression in VLMs](https://arxiv.org/abs/2603.14892).” 2026.
14. Ding et al. “[ET-Prune: Evidence-Aware Dynamic Budgeting for Visual Token Pruning in Text-Rich MLLMs](https://arxiv.org/abs/2608.01979).” 2026.
15. Wen et al. “[Not All Visual Tokens Are Equally Safe to Remove: Consequence-Sensitive Visual Token Compression](https://arxiv.org/abs/2608.09176).” 2026.
16. Ye et al. “[UReader: Universal OCR-free Visually-situated Language Understanding with Multimodal Large Language Model](https://arxiv.org/abs/2310.05126).” 2023.
17. Nassar et al. “[SmolDocling: An Ultra-Compact Vision-Language Model for End-to-End Multi-Modal Document Conversion](https://arxiv.org/abs/2503.11576).” 2025.
18. Li et al. “[ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use](https://arxiv.org/abs/2504.07981).” 2025.
19. Xu et al. “[SlowFast-LLaVA-1.5: A Family of Token-Efficient Video Large Language Models for Long-Form Video Understanding](https://arxiv.org/abs/2503.18943).” 2025.
20. Wang et al. “[mDPO: Conditional Preference Optimization for Multimodal Large Language Models](https://arxiv.org/abs/2406.11839).” 2024.
21. Allen Institute for AI. “[CoSyn-point Dataset Card](https://huggingface.co/datasets/allenai/CoSyn-point).” Accessed 2026.
22. Allen Institute for AI. “[CoSyn-400K Dataset Card](https://huggingface.co/datasets/allenai/CoSyn-400K).” Accessed 2026.
23. TensorFlow Datasets. “[RefCOCO Dataset Documentation](https://github.com/tensorflow/datasets/blob/master/docs/catalog/ref_coco.md?plain=1).” Accessed 2026.
24. Ling et al. “[OCRBench v2 Dataset Card](https://huggingface.co/datasets/ling99/OCRBench_v2).” Accessed 2026.
