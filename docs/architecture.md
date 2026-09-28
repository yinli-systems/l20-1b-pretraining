# Architecture and experiment lineage

## Language model

The released L20-1B-20B-Base has 22 decoder layers, width 2,048, 32 query heads,
4 key/value heads, a 5,632-wide SwiGLU MLP and a 2,048-token context. The model
and 32K tokenizer were trained from scratch. See the
[language model card](../pretraining/MODEL_CARD.md) for the bound configuration.

## Multimodal extension

The current combined model has 1,203,213,056 parameters. Its vision encoder is
an externally pretrained SigLIP2 base patch16 model; the combined VLM is not a
fully scratch-trained multimodal model.

| Path | Image processing | Visual tokens |
|---|---|---:|
| Low-resolution | 224px image → 14×14 patch grid | 196 |
| High-resolution T | 448px source resize → four native 224px tiles → reconstructed 28×28 raster grid | 784 |

The latest expansion freezes language and vision backbone weights, and trains
10,280,448 visual-bridge and attention-LoRA parameters. Decoder/low-vision
computation uses BF16, high-vision arithmetic uses the established FP32 path,
and trainable bridge/LoRA/Adam state remains FP32. These describe the experiment,
not a claim that the precision mix is globally optimal.

## Latest training mixture

Each 64-image update uses 32 newly streamed full images, 16 old natural-image
questions, 8 balanced full-image replays across four domains and 8 descriptions.
Text replay occurs every four updates. Earlier word-crop replay was replaced,
and existing multiple-choice targets were normalized to answer letters.
Consequently, the latest intervention changes more than dataset size.

## Separate experimental branches

The earlier vision-tail experiment trained the final two visual blocks. The
prefix-cache experiment reused only frozen-prefix features and kept trainable
blocks live. Neither should be confused with the frozen-vision streamed run.
Warm-cache gains were modest after cache-build cost, and are not serving-speed
or quality improvements. Historical compression/routing studies are preserved
in [the multimodal archive](../L20-VL-1.2B/HISTORY.md).

## Open questions, not conclusions

Higher-resolution document views, vision adaptation, better chart supervision
and a more mature language parent are plausible next axes. Current results do
not isolate any one of them as the limiting cause. In particular, the restricted
CoSyn data supplied chart-title transcription rather than general chart
arithmetic supervision; an unresolved ChartQA gain is not proof of an
architecture ceiling.
