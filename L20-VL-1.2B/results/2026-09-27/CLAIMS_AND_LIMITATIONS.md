# Claims and Limitations — 2026-09-27

## Supported claims

1. **Unique-image scaling improved local held-out multimodal performance.**
   Under the frozen local confirmation protocol, the final 1,934-step endpoint improved over the immutable parent by:
   - TextVQA-style ten-reference score: **+4.38 pp**, paired 95% CI **[+2.15, +6.91]**
   - DocVQA-style ANLS: **+3.43 pp**, paired 95% CI **[+1.15, +5.89]**
   - ChartQA-style relaxed accuracy: **+5.86 pp**, paired 95% CI **[+3.13, +8.98]**
   - AI2D strict letter accuracy: **+5.21 pp**, paired 95% CI **[-2.08, +13.02]**

2. **The final TextVQA-style endpoint depends more on the correct image than the earlier parent.**
   Final true-image score was **18.09%** versus **9.73%** after deterministic wrong-image substitution.

3. **Retention gates passed during the large-scale run.**
   The final old-QA probe was **56.25%**, caption NLL **1.61976**, and the text probes remained within the predeclared margins.

4. **The large-scale run is qualitatively different from the earlier small-pool continuation.**
   It used approximately **31k planned unique photos**, **30,958 train image clusters**, at most **two exposures per cluster**, and **61,888 new-image events**.

5. **A frozen-prefix training cache can provide a modest throughput improvement when images repeat.**
   Warm-cache throughput increased from about **19.04 to 20.10 img/s** on the bounded matched systems test.

## Claims explicitly NOT supported

The repository must not describe the current model as:

- state of the art;
- at SmolVLM parity;
- a released production VLM;
- fully pretrained from scratch as a multimodal model;
- validated on the full official TextVQA / DocVQA / ChartQA / AI2D leaderboards;
- proven to follow a population-level scaling law;
- proven to generalize across all document/OCR/chart distributions;
- proven to have eliminated benchmark contamination;
- proven to improve from vision-tail unfreezing alone;
- proven to gain quality from the prefix cache.

## Statistical boundary

The paired confidence intervals in the final scale experiment quantify uncertainty from the **evaluation image sample**, using paired cluster bootstrap resampling.

They do **not** quantify:

- training-seed variance;
- hyperparameter-search uncertainty;
- multiplicity across the four reported domains;
- source-dataset sampling uncertainty outside the constructed split.

Only one training seed was used in the largest run.

## Dataset boundary

The scale experiment uses local image-disjoint splits derived from training-source corpora. It verifies recorded image IDs, exact hashes, pixel hashes and registered near-image clusters against the project history, but does not claim exhaustive semantic or original-document deduplication.

TextVQA-style rows retain ten human references. DocVQA and ChartQA diagnostics in this campaign use single-reference local scoring. The AI2D diagnostic is strict answer-letter accuracy.

## SmolVLM comparison boundary

SmolVLM measurements are useful external references but are not matched-compute comparisons. Models differ in:

- language backbone maturity and pretraining exposure;
- visual tokenization;
- image processor behavior;
- parameters;
- visual token budget;
- precision;
- historical training exposure.

The source-capped SmolVLM diagnostic should not be described as an equal-architecture comparison.

## Systems boundary

The prefix cache:

- caches only the frozen high-vision prefix;
- leaves the trainable vision tail, bridge and language LoRA on the live gradient path;
- is training-only in the measured experiment;
- is not evidence of faster new-image serving;
- has only a modest amortized gain when cache construction is included.

## Release boundary

This repository stores code, protocols and evidence summaries only. Raw datasets, private checkpoints, optimizer states and model weights are intentionally excluded from the public Git history.
