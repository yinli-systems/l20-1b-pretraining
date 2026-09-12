---
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- causal-lm
- llama
- pretraining
- from-scratch
- single-gpu
- nvidia-l20
---

# L20-1B-20B-Base

L20-1B-20B-Base is a 1.10B-parameter English causal language model pretrained
from random initialization on 20.0B prediction tokens using one NVIDIA L20.
The tokenizer was also trained from scratch. This is a base completion model,
not an instruction/chat model, and no third-party pretrained checkpoint was
used to initialize it.

## Quick start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "AliceYin/L20-1B-20B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

prompt = "Machine learning is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
    )
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

The uploaded safetensors preserve the final checkpoint exactly in FP32. Loading
with BF16, as above, substantially reduces inference memory use.

## Architecture

| Field | Value |
|---|---:|
| Parameters | 1,100,048,384 |
| Architecture | LlamaForCausalLM |
| Layers | 22 |
| Hidden size | 2,048 |
| Attention heads / KV heads | 32 / 4 |
| MLP intermediate size | 5,632 |
| Context length | 2,048 |
| Vocabulary | 32,000 byte-level BPE |
| Positional encoding | RoPE, theta 10,000 |
| Weight tying | No |

## Training

The run completed 19,148 optimizer steps and 19,999,703,040 prediction tokens.
It used BF16 mixed precision, compiled PyTorch SDPA, fused AdamW, sequence length
2,048, micro-batch 6, gradient accumulation 85, and an effective global batch
of 510 sequences. The learning rate used 1,000 warmup steps followed by cosine
decay from 4e-4 to 4e-5.

The English training mixture was:

- 42.5% FineWeb-Edu-Dedup, score 4+
- 42.5% DCLM-Baseline
- 3% FineMath-4+
- 12% permissively licensed Stack-Edu code in Python, JavaScript, TypeScript,
  C++, and Java at the release-validated educational thresholds

Accepted documents passed English/content filters, normalized-text SHA-256
exact deduplication, and a 13-word benchmark decontamination check. DCLM email
addresses and syntactically valid IPv4 addresses were deterministically
anonymized. The original decontamination list did not explicitly include BoolQ;
results therefore include a six-task sensitivity analysis excluding BoolQ.

## Hardware, throughput, and MFU

Training used exactly one NVIDIA L20 with 46,068 MiB VRAM. The software snapshot
recorded NVIDIA driver 580.159.04, CUDA 13.0, PyTorch 2.12.1+cu130, Transformers
4.56.2, and lm-eval 0.4.9.

A representative live snapshot at step 10,595 recorded:

| Metric | Value |
|---|---:|
| Throughput | 12,845 tokens/s |
| Model FLOP/s | 93.948 TFLOP/s |
| Standard MFU | 71.17% |
| GPU utilization | 100% |
| VRAM use | 45,265 / 46,068 MiB |
| Power | 348.3 W |

MFU is calculated as `model_FLOP/s / 132e12`, using exactly 132 TFLOP/s as the
BF16 peak denominator. This is point-in-time training telemetry, not a full-run
average. A separate 60-step preflight at the selected micro-batch measured
12,586 tokens/s and 36.13 GiB peak allocated memory; micro-batch 8 was rejected
after an out-of-memory failure.

## Evaluation

Final held-out validation loss was **2.4247** (perplexity **11.2990**). External
evaluation used lm-eval 0.4.9, BF16, maximum context 2,048, and fixed seeds.

| Task | Shots | Metric | Score |
|---|---:|---|---:|
| ARC-Challenge | 0 | acc_norm | 33.62% |
| ARC-Easy | 0 | acc_norm | 64.14% |
| HellaSwag | 0 | acc_norm | 45.13% |
| OpenBookQA | 0 | acc_norm | 35.00% |
| PIQA | 0 | acc_norm | 69.59% |
| WinoGrande | 0 | acc | 52.17% |
| BoolQ | 0 | acc | 59.17% |
| LAMBADA OpenAI | 0 | acc | 46.26% |
| TruthfulQA MC2 | 0 | acc | 36.91% |
| MMLU | 5 | acc | 25.31% |
| GSM8K | 5 | flexible exact match | 1.67% |

The primary seven-task macro over HellaSwag, PIQA, WinoGrande, OpenBookQA,
ARC-Easy, ARC-Challenge, and BoolQ is **51.2601%**. Excluding BoolQ gives
**49.9411%**.

Under the same frozen seven-task protocol, this checkpoint:

- beats TinyLlama 1T by **+1.0182 percentage points**, paired task-stratified
  bootstrap 95% CI **[+0.1584, +1.8860]**;
- ties TinyLlama 1.5T: +0.0910 pp, 95% CI [-0.7771, +0.9745];
- ties TinyLlama 2T: -0.3019 pp, 95% CI [-1.1978, +0.5871];
- loses to TinyLlama 2.5T: -2.6317 pp, 95% CI [-3.4891, -1.8006].

The intervals cover benchmark-item sampling uncertainty only. They do not cover
training-seed variance and are not adjusted for multiple comparisons. The full
36-checkpoint study contains both wins and losses; it is a frozen candidate set,
not a global census or a claim of universal token-efficiency leadership.

Machine-readable [final benchmark metrics](https://huggingface.co/AliceYin/L20-1B-20B-Base/blob/main/evaluation/final-benchmarks.json),
the complete [36-checkpoint comparison](https://huggingface.co/AliceYin/L20-1B-20B-Base/blob/main/evaluation/efficiency-all-results.md),
and [GPU telemetry](https://huggingface.co/AliceYin/L20-1B-20B-Base/tree/main/training)
are included in this release.

![Compute-efficiency frontier](https://huggingface.co/AliceYin/L20-1B-20B-Base/resolve/main/assets/efficiency-frontier-clean.png)

## Limitations

- This base model is not instruction tuned, preference tuned, or safety tuned.
- It may hallucinate, repeat text, produce biased or harmful content, and should
  not be used as a factual authority or autonomous decision maker.
- It is English-focused, has a 2,048-token context window, and is weak on
  multi-step mathematics in the reported evaluation.
- MMLU 5-shot required left truncation for some examples at the 2,048-token
  context limit, so that score should be interpreted with care.
- Exact deduplication and benchmark matching cannot prove the absence of all
  semantic near-duplicates, paraphrases, personal information, or contamination.
- Dataset quality filters improve the input distribution but do not guarantee
  correctness of every training document or model output.

## Reproducibility and release integrity

The release contains three sharded safetensors files. Conversion verified exact
equality for all 201 tensors against the final Transformers checkpoint. The
release manifest records file hashes, source checkpoint hash, parameter count,
and storage dtype. Training code, compact receipts, derived metrics, confidence
intervals, and plotting scripts are maintained in the public
[GitHub evidence repository](https://github.com/yinli-systems/l20-1b-pretraining).
