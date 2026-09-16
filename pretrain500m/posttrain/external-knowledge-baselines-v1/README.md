# Matched external closed-book SciQ baselines v1

This evaluation scores four public base language models on exactly the same 489
development rows, frozen choice order, prompt, and likelihood metrics as the
529M Base/F2 independent knowledge proxy. The SciQ `support` field is omitted.

Each model and tokenizer snapshot is pinned to a Hugging Face commit and checked
against a local per-file SHA-256 manifest before GPU loading. OpenELM uses the
officially documented Llama 2 tokenizer, pinned separately. All model loading is
offline and uses BF16 eager attention with deterministic PyTorch algorithms.

The primary metric is `acc_norm`: accuracy after choosing the answer with the
highest mean conditional log likelihood over its answer tokens. `acc` based on
total answer-token log likelihood is secondary. Tokenization necessarily differs
between model families, but prompt text, completion text, rows, choices, scoring
formula, dtype, batch size, and tie breaking are fixed.

These are development results on a public benchmark. They provide a matched
comparison for this one closed-book English science proxy, not broad model
superiority or a sealed test result. The reserved 490 confirmation rows remain
unscored.

## Completed result

ParaCloud array job 1594569 completed all four candidate tasks and every job,
batch, and extern record exited `0:0`. The primary `acc_norm` results are:

| Model | Parameters | Correct | `acc_norm` | Delta vs. 529M Base | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM2-360M | 361,821,120 | 380/489 | 77.7096% | +16.3599 pp | [+12.0654, +20.6544] pp |
| Qwen2.5-0.5B | 494,032,768 | 310/489 | 63.3947% | +2.0450 pp | [-2.2495, +6.3395] pp |
| 529M Base | about 529M | 300/489 | 61.3497% | reference | - |
| MobileLLM-600M | 603,188,352 | 272/489 | 55.6237% | -5.7260 pp | [-10.8384, -0.6135] pp |
| OpenELM-450M | 457,179,136 | 124/489 | 25.3579% | -35.9918 pp | [-41.9223, -30.2658] pp |

The 529M Base therefore beats MobileLLM and OpenELM under this exact protocol,
is statistically unresolved against Qwen on the primary metric, and is clearly
behind SmolLM2. The raw result rows were independently rebound and rescored, all
pinned snapshot files passed SHA-256 verification, and the 10,000-resample
paired intervals reproduced exactly. The compact result receipt and evidence
snapshot are in
[`../../result-archive/external-knowledge-baselines-v1-development-results-20260916.json`](../../result-archive/external-knowledge-baselines-v1-development-results-20260916.json)
and
[`../../result-archive/para-external-knowledge-baselines-v1-20260916/`](../../result-archive/para-external-knowledge-baselines-v1-20260916/).

The requested official MobileLLM PyTorch weights were gated for the current
account. Its score uses a strict PyTorch reconstruction of a pinned public
full-precision ONNX export that declares `facebook/MobileLLM-600M` as its base.
All keys and shapes loaded strictly and every parameter was finite, but direct
weight-hash identity against the gated official file could not be established.
