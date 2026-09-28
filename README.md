# L20 Pretraining Lab

**From a scratch-trained 1.1B language model to controlled multimodal experiments — on one NVIDIA L20.**

[![CI](https://github.com/yinli-systems/l20-1b-pretraining/actions/workflows/ci.yml/badge.svg)](https://github.com/yinli-systems/l20-1b-pretraining/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![Language weights](https://img.shields.io/badge/Hugging_Face-L20--1B--20B--Base-yellow.svg)](https://huggingface.co/AliceYin/L20-1B-20B-Base)

[Latest results](results/2026-09-28/README.md) · [Language model](pretraining/README.md) · [Multimodal research](L20-VL-1.2B/README.md) · [Reproduce the evidence](docs/reproducibility.md)

## Two tracks, one evidence trail

| Track | What is available | Status |
|---|---|---|
| **L20-1B-20B-Base** | A 1.100B language model and 32K tokenizer trained from scratch; 19,999,703,040 prediction tokens; code and hash-bound evaluation records | Language weights released |
| **L20-VL-1.2B** | The same language base with an externally pretrained SigLIP2 encoder; 1.203B total parameters, 10.28M trainable bridge/attention-LoRA parameters in the latest run | Research endpoint; **not promoted** |

The language base reaches **51.2601%** on its frozen seven-task zero-shot macro
(**49.9411%** without BoolQ). Task definitions, exclusions and the complete
36-checkpoint comparison are in the [language model card](pretraining/MODEL_CARD.md).
The combined vision-language model is **not** fully pretrained from scratch.

## Latest multimodal result · September 28, 2026

A streamed expansion completed **5,120 updates** and **163,840 new-image
training events**. The data pipeline prepared 165,376 images; prepared data
are not completed updates, and the 250,000-image cap was not reached.

The same **15,937 questions** were evaluated before and after training, using
complete published validation/test splits and a custom model wrapper aligned
to pinned task/scoring definitions — not a full `lmms-eval` CLI run or a
leaderboard submission.

| Task | Before | After | Change | Paired 95% CI, pp |
|---|---:|---:|---:|---:|
| TextVQA · 5,000 questions | 17.51 | **23.77** | **+6.27 pp** | [+5.35, +7.16] |
| DocVQA · 5,349 questions | 4.59 | **9.48** | **+4.89 pp** | [+4.22, +5.56] |
| ChartQA · 2,500 questions | 11.32 | **12.16** | +0.84 pp | [−0.16, +1.89] |
| AI2D · 3,088 questions | 7.32 | **26.26** | +18.94 pp | [+17.39, +20.60] |

**The trade-off matters:** old natural-image QA fell from **56.25% to 50.86%**.
The last two checks exceeded the predeclared 5-point drop allowance, so the
retention guard stopped training. This endpoint is retained for analysis, **not
promoted as an improved all-purpose release**.

AI2D gains include output-format repair; ChartQA's interval crosses zero.
Intervals measure evaluation-image sampling, not training-seed variation.
Two known ChartQA overlap questions remain disclosed and included. Source mix,
replay and target formatting changed with scale, so this is **not a data-only
causal scaling law**. No frontier or SmolVLM-parity claim is made.

**Inspect:** [full report](results/2026-09-28/README.md) ·
[metrics](results/2026-09-28/metrics.json) ·
[aggregate evidence](results/2026-09-28/evidence/) ·
[paired image-cluster counts](results/2026-09-28/scores/)

## Verify without a GPU

```bash
git clone https://github.com/yinli-systems/l20-1b-pretraining.git
cd l20-1b-pretraining
python3 tools/verify_repository.py
python3 pretraining/reproducibility/recompute.py
```

The first command verifies the repository migration, public result hashes,
paired score totals and active documentation links. The second runs the
**unchanged** language-model evidence verifier. Neither regenerates model
predictions. See [reproducibility](docs/reproducibility.md) for tests and the
boundary between numerical replay and end-to-end training reproduction.

## Repository guide

```text
pretraining/       Language training, evaluation, original reports and tests
L20-VL-1.2B/       Multimodal implementation, protocols and earlier experiments
results/           Dated result reports, metrics and score evidence
docs/              Architecture, limitations, reproduction and migration guide
tools/             Fast CPU-only verification
tests/             Repository and latest-result regression checks
.github/           Continuous integration and contribution checklist
```

The reorganization preserves **all 646 previously tracked files byte-for-byte**
at the paths recorded in the [migration manifest](docs/layout-migration.json).
Historical negative results are retained, not removed to simplify the story.

## Research boundaries

The released text model is a base completion model, not a safety-tuned chat
assistant. Multimodal checkpoints are not included in this public repository.
Source images, raw datasets, optimizer states and credentials are excluded.
The latest VLM's unresolved weaknesses include document reading, chart reasoning
and retention; possible remedies remain hypotheses rather than established
architecture diagnoses. See [limitations](docs/limitations.md).

## License

Repository-authored code and documentation: [MIT](LICENSE). External data,
pretrained encoders, baseline models and generated model artifacts retain their
own licenses and terms. Part of [Pretraining Lab](https://github.com/yinli-systems/pretraining-lab).
