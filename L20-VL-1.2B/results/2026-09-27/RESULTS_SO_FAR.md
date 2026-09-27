# L20-VL-1.2B — Results So Far (2026-09-27)

This document consolidates the current experimental lineage for the 1.2B-class VLM built on the scratch-trained L20-1B language model. It reports positive, negative, and inconclusive results. It intentionally excludes model weights, raw datasets, private checkpoints, and any claim of public-leaderboard parity.

## 1. Language-model parent

- Language parameters: **1.100048384B**
- Pretraining exposure: **~20.0B tokens**
- Context: 2048
- Seven-task macro: **51.2601%**
- Six-task macro excluding BoolQ: **49.9411%**
- Parent artifact: `L20-1B-20B-Base`
- Combined VLM with SigLIP2 is **not** fully pretrained from scratch because the vision encoder is externally pretrained.

## 2. Core VLM architecture

- Combined parameters: **1,203,213,056**
- Main trainable adapter path before vision-tail experiments: **10,280,448 parameters**
- Vision encoder: SigLIP2 base patch16
- Default 224px path: **196 visual tokens**
- High-resolution 448px path: **784 visual tokens**
- Language base frozen in the main adapter experiments
- Attention LoRA: rank 16, alpha 32
- Bridge and LoRA weights/optimizer kept in FP32; backbone compute BF16 unless otherwise noted

A 49-token compression path was tested earlier and did not replace the 196-token path for the current general-vision lineage.

## 3. General-vision lineage before the scale-up

The selected general-vision lineage ran:

- **1,024 updates**
- **65,536 image events**
- **594,278 supervised answer tokens**
- **14,663,423 non-padding image+text input tokens**
- **4,194,304 text replay tokens**
- **40.656 img/s**
- **13.30 GiB** peak allocated memory

On the frozen local VQAv2-style official-validation subset:

- true-image consensus: **45.8594%**
- wrong-image consensus: **32.598%**
- correct-image advantage: **+13.2617 pp**
- complementary true score: **38.008%**
- pair-joint exact: **10.156%**

These are subset measurements, not full official leaderboard scores.

## 4. Same-machine SmolVLM gap audit

Measured `HuggingFaceTB/SmolVLM-500M-Instruct` at revision
`a7da5b986cb59b408707209984f360a5f4ad7e47`.

On the same local VQA subset:

| Model / route | Score |
|---|---:|
| L20-VL selected lineage | 45.86% |
| SmolVLM native | 71.78% |
| SmolVLM source-capped / reduced-token diagnostic | 65.90% |

The resulting gap was approximately **-25.92 pp** versus native and **-20.04 pp** versus the source-capped diagnostic.

Latency on a separate real-image batch-1 diagnostic:

- L20-VL first-token p50: **37.74 ms**
- SmolVLM native first-token p50: **346.16 ms**

This is **not** a same-work speedup claim: the visual-token budgets and processor behavior differ materially.

## 5. Negative and inconclusive scaling experiments

### 5.1 Expanded natural-image data and MLP-LoRA

Three-seed 256-step study:

- Control C mean: **0.50690**
- Expanded-data D mean: **0.48997**
- Expanded-data + MLP-LoRA M mean: **0.48867**

Neither D nor M passed the development gate. The MLP-LoRA extension was not promoted.

### 5.2 Partial replay mixture

Three-seed 512-step study:

- C mean: **0.52201**
- Replay R mean: **0.50182**
- R-C per seed: **[-0.02656, -0.04570, +0.01172]**

The replay arm did not pass the multi-seed gate and was not promoted.

These negative results are kept because they ruled out "more examples per update" and "more adapter parameters" as sufficient explanations for the gap.

## 6. V2 multi-domain / dual-resolution pilot

A bounded pilot introduced AI2D, ChartQA, DocVQA and TextVQA-style supervision, with matched image-update budgets and 224/448 routes.

On tiny local held-outs, 448px supervision reduced non-EOS NLL substantially relative to the parent:

- AI2D: **1.3108 -> 0.7927**
- ChartQA: **3.9408 -> 2.3586**
- DocVQA: **4.0901 -> 3.0180**
- TextVQA: **4.7175 -> 3.0172**

Because the held-outs were tiny and generation accuracy remained weak, this was treated only as directional evidence.

## 7. Reading/generalization and full-image integration

A later full-image integration compared two equal-token 448px routes:

- G: interpolated full448
- T: four native224 tiles restored into one ordered 28x28 patch grid

Both use **784 visual tokens**.

On a 128-image official-validation subset:

| Endpoint | Ten-reference score |
|---|---:|
| parent-G | 5.70% |
| parent-T | 6.25% |
| G | 6.95% |
| T | **8.83%** |
| SmolVLM source448 | 41.02% |
| SmolVLM native | 57.50% |

The 448px integration improved local training-source diagnostics more than external validation and therefore was not treated as parity evidence.

## 8. Vision-tail and prefix-engine study

The final two high-resolution vision blocks were unfrozen in controlled arms.

On a fresh 256-image TextVQA validation subset:

| Model | Ten-reference consensus |
|---|---:|
| parent | 9.96% |
| C: frozen high vision continuation | 8.71% |
| V: last two blocks, LR 1e-6 | 8.32% |
| V10: last two blocks, LR 1e-5 | 9.22% |
| P: +128 updates | 9.18% |
| SmolVLM source448 | **44.49%** |
| SmolVLM native | **61.80%** |

The strongest diagnostic signal was a severe train-vs-unseen gap:

- V10 known TRAIN images: **74.45%**
- V10 fresh official images: **9.22%**
- P known TRAIN images: **84.38%**
- P fresh official images: **9.18%**

This falsified the hypothesis that simply unfreezing the vision tail and adding more updates on the small pool would close the gap.

### Prefix-cache systems result

A training-only cache reuses the frozen vision prefix while leaving the last two trainable blocks live.

- baseline: **19.04 img/s**
- warm cache: **20.10 img/s**
- warm-cache gain: **+5.61%**
- three-pass amortized gain including initial cache construction: **+0.79%**

Full-gradient checks covered **215 trainable tensors** and matched on the tested batches. A later eviction robustness revision also passed real-GPU gradient checks. This is a modest systems optimization, not a model-quality result or serving-speedup claim.

## 9. Large unique-image scaling run

The largest current run changed the main experimental variable from repeated exposure to **unique multimodal image coverage**.

### Data / sampling

- admitted unique active images across splits: **39,526**
- train unique photos: **31,469**
- train image clusters: **30,958**
- unique photos in the actual two-pass plan: **31,178**
- unique training examples in the plan: **47,690**
- new image events: **61,888**
- global image events: **123,776**
- maximum new-image-cluster exposure: **2**
- training steps: **1,934**
- trainable parameters: **10,280,448**
- training time in the main process: **6,430.38 s (~1.79 h)**

New-task image events by domain:

- TextVQA: **25,950**
- ChartQA: **25,978**
- DocVQA: **6,498**
- AI2D: **3,462**

The final checkpoint was frozen before accessing the confirmation results.

## 10. Final image-disjoint confirmation result

The final confirmation uses local image-disjoint splits originating from training-source corpora. It is **not** a full public benchmark result.

| Task | Parent | Final | Delta | Paired 95% CI |
|---|---:|---:|---:|---:|
| TextVQA, 256 images | 13.71% | **18.09%** | **+4.38 pp** | **[+2.15, +6.91]** |
| DocVQA, 256 images | 4.79% | **8.22%** | **+3.43 pp** | **[+1.15, +5.89]** |
| ChartQA, 256 images | 7.81% | **13.67%** | **+5.86 pp** | **[+3.13, +8.98]** |
| AI2D, 192 images | 19.79% | **25.00%** | +5.21 pp | [-2.08, +13.02] |

The first three task intervals are entirely above zero for **evaluation-sample uncertainty**. They do **not** measure training-seed uncertainty, and the four-domain comparisons are exploratory rather than multiplicity-adjusted confirmatory claims.

### Correct-image dependence at the final endpoint

| Task | Correct image | Wrong image | Advantage |
|---|---:|---:|---:|
| TextVQA | 18.09% | 9.73% | **+8.36 pp** |
| DocVQA | 8.22% | 5.45% | **+2.77 pp** |
| ChartQA | 13.67% | 8.59% | **+5.08 pp** |
| AI2D | 25.00% | 23.96% | +1.04 pp |

The AI2D grounding signal remains weak even though its point estimate improved.

### Retention at the final endpoint

- old natural-image QA: **56.25%**
- old caption NLL: **1.61976**
- text NLL:
  - web: 2.38753
  - DCLM: 2.71388
  - math: 2.39383
  - code: 1.02075
- consecutive retention-gate failures: **0**

## 11. What the current evidence supports

The strongest current conclusion is:

> Increasing unique multimodal image coverage from the earlier small repeated pool to ~31k training photos produced measurable unseen gains on TextVQA, DocVQA and ChartQA under the frozen local confirmation protocol.

The current evidence does **not** support:

- SmolVLM parity
- state-of-the-art claims
- a full official benchmark score
- a claim that 448px alone caused the gains
- a claim that the frozen-prefix cache improves quality
- a claim that AI2D grounding is solved
- a claim that one training seed establishes a population-level scaling law

## 12. Next research questions

1. Scale unique multimodal data further while preserving image/document-disjoint evaluation.
2. Increase underrepresented DocVQA / AI2D coverage without simply oversampling the same images.
3. Separate data-scale gains from vision-tail adaptation with matched multi-seed experiments.
4. Revisit language-backbone maturity as an independent controlled axis.
5. Add broader OCR/document/chart benchmarks and hallucination controls.
6. Continue reporting wrong-image controls and non-EOS content NLL separately from free-generation accuracy.

See `metrics.json` for machine-readable headline values and `CLAIMS_AND_LIMITATIONS.md` for the reporting boundary.
