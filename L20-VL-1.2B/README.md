# L20-VL-1.2B

**A research extension of the scratch-trained L20 language base with an externally pretrained SigLIP2 encoder.**

[Latest full-split result](../results/2026-09-28/README.md) · [Earlier results](results/2026-09-27/RESULTS_SO_FAR.md) · [Architecture](../docs/architecture.md) · [Historical protocols](HISTORY.md)

## Current endpoint

The September 28 streamed expansion completed 5,120 updates and 163,840 new-image
training events. Complete published-split evaluation gives **23.77 TextVQA,
9.48 DocVQA, 12.16 ChartQA and 26.26 AI2D** on a 0–100 scale.

It also reduced old natural-image QA from 56.25% to 50.86%, triggering the
retention guard. **The model is not promoted or released.** AI2D gains include
format normalization, and ChartQA does not show a statistically resolved gain.
The [full report](../results/2026-09-28/README.md) contains denominators,
confidence intervals, score files and limitations.

## Architecture used in the latest run

```text
Image → frozen SigLIP2 → trainable visual bridge → language decoder + attention LoRA
                           196 or 784 visual tokens
```

The language and vision backbones remain frozen in this run. Trainable bridge
and attention-LoRA parameters total 10,280,448. A separate vision-tail study
exists but did not initialize the final streamed expansion.

## Find the earlier work

| Area | Entry point |
|---|---|
| General-vision, replay, MLP-LoRA and dual-resolution studies | [September 27 result archive](results/2026-09-27/RESULTS_SO_FAR.md) |
| Routing, compression, Stage A/D and original gates | [Historical overview](HISTORY.md) |
| Experimental implementations | [Source files in this directory](.) |
| Original evidence receipts | [evidence/](evidence/) |
| Research plans and claim boundaries | [research/](research/) |

Old proposals describe their historical stage, not the current training state.
Positive, negative and inconclusive results all remain available.

## Reproduction boundary

This public tree is an implementation and evidence archive. It does **not**
contain the final VLM weights, raw images, optimizer state or every machine-local
runtime dependency from the newest campaigns. Do not interpret it as a
one-command release of the current VLM. Numerical checks are available without
those artifacts through the [repository verification guide](../docs/reproducibility.md).
