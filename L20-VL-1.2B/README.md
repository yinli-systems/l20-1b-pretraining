# L20-VL-1.2B

This is an isolated, fail-closed research workspace for testing whether the
released 1.100B language Base can be extended with a compact pretrained vision
encoder on one NVIDIA L20. It contains protocols, code, and evidence receipts,
but no model weights or raw data. The current system is not a released or
deployment-ready multimodal model.

## Evidence-adaptive routing results

The current architecture starts from a 49-token global view, predicts either a
14x14 evidence patch or `STOP`, and acquires at most one source-resolution crop.
The complete research rationale, literature boundary, metrics, and stop rules
are in
[`research/evidence-adaptive-multimodal-roadmap-v2-20260914.md`](research/evidence-adaptive-multimodal-roadmap-v2-20260914.md).

| Evaluation | Result | Decision |
|---|---:|---|
| Synthetic confirmation, POINT/STOP | 100.00% | Passed |
| Synthetic confirmation, exact patch | 96.09% | Passed |
| Synthetic predicted-crop local answer path | 96.48% row / 95.31% family-joint | Passed |
| Open Images adaptation, POINT/STOP | 98.96% | Passed component gate |
| Open Images adaptation, point inside official box | 80.73% | Passed component gate |
| Open Images adaptation, within one patch of box center | 56.25% | Failed frozen 70% gate |
| Synthetic retention after Open Images adaptation | 100.00% exact / within-one / STOP | Passed |
| Unmarked Open Images referring-expression audit | 32/48 accepted | Rejected for data noise |

The Open Images adaptation updated only the 376,706-parameter router. Its
960-step cached optimizer phase took 15.32 seconds on one NVIDIA L20 at
6,015.64 rows/s and 2.41 GiB peak allocated memory; these numbers exclude
feature caching and frozen-model loading. Relative to the frozen synthetic
parent, the selected diagnostic checkpoint improved official-box hit rate by
44.27 percentage points, with paired bootstrap 95% CI `[+34.90, +53.13]`.
However, no checkpoint passed every predeclared natural-development gate, so no
natural confirmation split was accessed and no natural-grounding success is
claimed.

See
[`evidence/openimages-evidence-router-adaptation-v1-summary.json`](evidence/openimages-evidence-router-adaptation-v1-summary.json)
for the checkpoint-level metrics, artifact hashes, and claim boundary. The
negative unmarked-data audit is retained because official annotation uniqueness
did not guarantee visible uniqueness or semantically correct generated prompts.

The research target is now frozen as **Counterfactual Evidence-Preserving
Visual Compression**: determine whether an explicit cross-image
answer-preference-delta objective preserves decision-relevant evidence at the
same 49-token budget better than ordinary distillation plus counterfactual
supervision and invariance. The hypothesis is unproven. The complete novelty
boundary, strong-baseline matrix, statistics, and stop rules are in
`research/counterfactual-evidence-preserving-compression.md` and
`counterfactual_protocol.json`.

The remote execution workspace contains the pinned SigLIP2 weight artifact;
weights remain excluded from Git. Its official-Hub SHA-256 was independently
matched after transfer, and the frozen BF16 forward smoke passed. The NVIDIA
synthetic OCR components `ocr_1` and `ocr_3` passed file-integrity checks but
were rejected for training after manual review found pseudo-English and
gibberish labels.

PixMo-Cap's four frozen metadata shards were hash-verified and all 717,042 rows
were audited. A deterministic 288-image sample from nine AI2-hosted families
decoded completely, but manual review found family-specific factuality,
formatting, privacy, rights, and benchmark-overlap blockers. Those 288 images
are authorized only for forward-only systems testing; no source is admitted to
formal training.

The real-image Stage-0 test passed through image decoding, SigLIP2, all four
bridge compression arms, and the released Base with zero backward calls and
zero training tokens. Target 1x/4x/9x/16x measured 72.1/97.9/103.3/108.9
images/s and 14.7k/19.9k/21.0k/22.2k non-padding caption tokens/s. Parent and
bridge hashes were unchanged. These are forward-only systems numbers, not
training throughput, MFU, or multimodal quality evidence.

A deterministic controlled diagnostic has also been generated: 5,000 scene
families, 15,000 globally unique images, five balanced task strata, and
scene-family-atomic 70/15/15 splits. Its manifest SHA-256 is
`df65dbef20fb41d8d08f76a14fb9d3bf5f778a82c2210e56287cb8c221508d7e`.
Two earlier builds were deliberately rejected and preserved—one for an exact
image collision and one for relation-edit occlusion. The corrected build passed
a 60-family/180-image contact-sheet audit, but is admitted only for controlled
diagnosis after the full-token foundation passes. It does not authorize model
training or support a natural-image claim.

`modeling.py` includes learned-query and parameter-free 2D spatial-pooling
compressors. `counterfactual_losses.py` implements reliability-gated preference
delta and invariance losses plus clustered bootstrap primitives.
`evaluate_counterfactual_predictions.py` evaluates frozen JSONL predictions
with paired scene-family bootstrap confidence intervals.

`protocol.json` freezes the current architecture candidate, immutable upstream
revisions, audit requirements, pilot caps, evaluation candidates, stage gates,
and stop rules. `source_registry.json` records metadata-only data candidates;
every source remains blocked. `quality_gates.json` freezes paired clustered-CI,
true/no/random-image controls, multi-seed confirmation, and text-retention
requirements before seeing pilot results. Run both `python3 validate_protocol.py`
and `python3 validate_sources.py`, plus
`python3 validate_counterfactual_protocol.py`, before any project action.

The language model and tokenizer were pretrained from random initialization.
The proposed SigLIP2 vision encoder was pretrained separately; therefore the
combined model must not be described as fully pretrained from scratch.

## Stage D controlled replication

Stage C produced a real but not yet causal observation: the selected 49-token
answer-only checkpoint scored 75.733% versus 73.333% for the selected 196-token
checkpoint on 3,000 controlled test families, while its candidate-scoring GPU
path was 2.778x faster. The 49-token branch, however, received an additional
875-step continuation and has a 4,726,274-parameter spatial-query compressor.
That comparison therefore does not isolate compression.

Stage D first audits the scorer and then runs two matched continuations:
`F_matched_196` and `A_spatial_49`. Both start from the same selected parent,
use the same 14,000 training families, receive the same LR-search allowance,
updates, seed pairing and checkpoint-selection opportunities, and use only
answer supervision. This remains a package-level comparison; a later
`spatial_196` mechanism control is required to isolate token count from
compressor capacity and layout.

The D0 scorer contract reports family-joint and ordinary per-query metrics,
valid-answer rates and answer distributions. FP32 scoring with TF32 disabled
passed exact rerun, batch-size, candidate-order and manifest-order invariance,
plus answer-span and image-hash gates. The earlier BF16 batch mismatch is
preserved as a negative receipt.

Fresh Stage D data contain 2,000 development, 3,000 sealed IID-test and 2,000
sealed OOD-binding families. The OOD partition separates binding-only swaps
from binding-plus-boundary-geometry changes. The accepted v2 build has 21,000
unique rendered images, zero exact overlap with Stage C, programmatic
visibility/non-overlap gates, and a 144-family human contact-sheet audit. No
Stage D test inference is authorized during LR selection or five-seed training.

See `evidence/STAGE_D_D0_AUDIT.md`, `research/stage-d-metric-contract.md`,
`stage_d_data_protocol_v2.json`, and `stage_d_lr_screen_protocol.json`. D0 and
the LR screen are preflight engineering evidence, not multi-seed efficacy or
natural-image evidence.

The audited 150-step LR screen selected `F_matched_196=0.5x` and
`A_spatial_49=1.0x` from the frozen three-rate grid, using 0.609 total L20
GPU-hours. The two-arm five-seed D1 protocol is now frozen at SHA-256
`dd04741069c4be356156b61dfba9105bccbcdb7efe78030a08a996b546f86d5e`.
Its five adaptation seeds are disjoint from the LR-selection seed, and all ten
checkpoints must be selected and frozen before any new test inference. See
`evidence/STAGE_D_LR_SCREEN.md` and `stage_d_replication_protocol.json`.

## Stage-A full-token foundation

The user has authorized a bounded data audit and, only after every gate passes,
up to 12 L20 GPU hours for Stage-A training. The frozen plan combines official
CLEVR v1.0 with a strictly filtered, attribution-preserving Open Images
Localized Narratives subset. See `research/stage-a-data-plan.md`,
`stage_a_data_admission.json`, and `stage_a_training_protocol.json`.

Acquisition does not automatically start training. The source archive receipt,
automated content audit, 200-image human audit, and exact training-manifest
hashes must all pass first. `train_stage_a_full_token.py` also fails closed
unless the human-audit decision explicitly admits the research pilot. The
resulting model, if trained, remains a conventional 196-token foundation and
does not itself test the proposed compression method.
