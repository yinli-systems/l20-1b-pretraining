# 500M English base-model pretraining: research and frozen protocol

Status: pretraining design and evaluation protocol frozen before the formal run. Systems and learning pilots are reported separately from model quality. The fixed operational deadline is 2026-09-13 22:20:31 UTC.

## Decision

The formal candidate is a decoder-only, English base model initialized from scratch: 26 layers, width 1,280, SwiGLU width 3,584, 20 query heads, 5 key/value heads, 64-dimensional heads, RoPE, RMSNorm, native grouped-query SDPA, tied input/output embeddings, and a 2,048-token context. With the fixed 50,280-token OLMo tokenizer it contains exactly **528,748,800 trainable parameters**.

This deep, comparatively thin shape follows two consistent public signals. MobileLLM identifies depth over width, embedding sharing, and GQA as effective design choices for sub-billion-parameter models; its 600M model uses 40 layers at width 1,152. SmolLM/SmolLM2 also adopt MobileLLM-like deep/thin shapes with GQA and tied embeddings for their smaller models. These sources establish a design prior rather than performance for this implementation. See the [MobileLLM paper](https://arxiv.org/abs/2402.14905), [MobileLLM repository](https://github.com/facebookresearch/MobileLLM), and [SmolLM training report](https://github.com/huggingface/blog/blob/main/smollm.md).

The selected corpus is the immutable `sample-10BT` snapshot of FineWeb-Edu at revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. All 14 Parquet files were checked against upstream hashes and footers before use. FineWeb is cleaned and deduplicated English Common Crawl data; FineWeb-Edu applies an educational-quality classifier, and the paper reports stronger knowledge/reasoning results than the parent web corpus. The primary source is the [FineWeb paper](https://arxiv.org/abs/2406.17557) and the released [FineWeb dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb).

The packer admits nonempty English rows with `language_score >= 0.90` and `int_score >= 3`, removes exact normalized-document duplicates, excludes a whole document if any normalized 13-word span matches the frozen benchmark database, and assigns validation by a deterministic document-hash rule. It writes independent 2,049-token blocks: 2,048 inputs and the exactly one-token-shifted targets. No raw shared corpus file is changed or deleted.

The target pack contains 8,999,998,914 raw packed tokens, corresponding to 8,995,606,528 unique prediction tokens. The formal optimizer budget is 15,999,172,608 prediction tokens, or **1.778554 data epochs**. Repetition is declared rather than hidden. The data-constrained scaling study found little loss penalty through roughly four repeated epochs at fixed compute; that result supports this amount of repetition but does not prove equal downstream accuracy for this corpus or model. See [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264).

## Data-recipe research

[DataDecide](https://arxiv.org/abs/2504.11393) is the most relevant controlled public evidence: 25 data recipes, 14 model scales, three seeds, and models up to 1B parameters. Its authors report that small single-scale rankings predict about 80% of pairwise choices at larger scale and that continuous likelihood proxies can exceed 80% predictability at a small fraction of target compute. This justifies treating their controlled 530M results as a data-selection prior, while still requiring our own frozen evaluation.

An immutable local scan of the released DataDecide evaluation Parquets found the following three-seed, seven-task OLMES means for `DCLM-Baseline (QC 7%, FW3)` at 530M: 0.47922 at 2.294B tokens, 0.50350 at 4.588B, 0.52821 at 10.322B, 0.54422 at 19.497B, and 0.57811 at 47.022B. It led the inspected recipes at several relevant scales and was the first-choice data candidate. The 21-shard, approximately 36.1GB candidate download failed before any shard completed because the ParaCloud login host returned `Network is unreachable`. Moving that volume through the slower local SSH link would consume too much of the fixed deadline. The formal corpus therefore uses the already downloaded, hash-verified FineWeb-Edu snapshot. DataDecide numbers remain **upstream OLMES context**; their prompts, implementation, and token budgets are not our frozen lm-eval protocol.

## Public comparison context

| Model | Architecture/data fact | Why it is not automatically comparable |
|---|---|---|
| DataDecide 530M | 16 layers, width 1,344, untied embeddings; tabled peak LR 2.8e-3 and 53B-token full budget ([model card](https://huggingface.co/allenai/DataDecide-dclm-baseline-qc-7p-fw3-530M)) | OLMES evaluation and a much larger token budget |
| SmolLM2-360M | 32 layers at width 960; model card reports 4T training tokens ([model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)) | Smaller model, over 200 times this run's tokens, different mixture and protocol |
| MobileLLM-600M | 40 layers at width 1,152, GQA, shared embeddings; card reports 1T tokens ([model card](https://huggingface.co/facebook/MobileLLM-600M)) | Different tokenizer, corpus, compute and evaluation harness |
| OpenELM-450M | 20 layers with layer-wise scaling; about 1.8T reported training tokens ([model card](https://huggingface.co/apple/OpenELM-450M)) | Different architecture, data mixture and evaluation settings |
| Qwen3-0.6B-Base | 28 layers, width 1,024, 16 query/8 KV heads, 151,936 vocabulary ([base config](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json)) | Qwen3 family training uses orders of magnitude more tokens and proprietary mixture components |
| Falcon-H1-0.5B-Base | Hybrid Transformer/Mamba model ([model card](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base)) | Hybrid architecture, large curated mixture and different reported benchmark protocol |

Published scores from these cards are useful orientation only. A direct claim requires rerunning the exact base checkpoints under `protocol-v2.json`; post-trained or instruct checkpoints are excluded.

## Optimizer and schedule evidence

The shared points across relevant recipes are Adam/AdamW, beta1 0.9, beta2 0.95, BF16, gradient clipping, and warmup followed by cosine or trapezoidal/WSD decay. MobileLLM reports Adam, weight decay 0.1, initial LR 2e-3, and cosine decay in its controlled sub-billion experiments. The public SmolLM2-360M pretraining config uses LR 3e-3, betas 0.9/0.95, 2,000 warmup steps, a long stable phase, and linear cooldown. DataDecide's 530M table uses LR 2.8e-3. Sources: [MobileLLM paper](https://openreview.net/pdf?id=EIGbXbxcUQ), [SmolLM2 config](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/smollm2/config_smollm2_360M.yaml), and [DataDecide model card](https://huggingface.co/allenai/DataDecide-dclm-baseline-qc-7p-fw3-530M).

Because these recipes disagree and the available budget is only 16B tokens, the peak LR is selected by fixed-data real-GPU pilots. A 30-step screen eliminated 3.5e-3 for early instability and worse validation loss. The remaining 1.5e-3 and 2.5e-3 candidates run for 300 steps on the same data order and held-out split. Before those longer results, `lr-selection-rule-v2.json` was frozen at SHA-256 `8d2e0ef5c8def7c14ebbdf5e52d056539944b178cbbe53d369611d5dbbe8320d`: candidates must remain finite and above 65% median MFU, then the lower mean validation loss at steps 100, 200, and 300 wins, with lower LR as the tie-breaker. That pilot is an optimization diagnostic, not benchmark selection. Formal settings otherwise stay fixed: AdamW, betas 0.9/0.95, epsilon 1e-8, weight decay 0.1, global batch 2,097,152 prediction tokens, 1.0 gradient clipping, short linear warmup, stable phase, and cosine final 10% decay.

## Systems evidence and MFU boundary

The allocated devices were verified in-job as RTX 5090 cards with 32,607 MiB each, driver 580.82.07, and PCIe-only topology. `torch.compile` BF16 synthetic training for the exact candidate on 8 RTX 5090 GPUs, two nodes, microbatch 4, and accumulation 32 measured 323,213.6 token/s and 20.13 GiB peak allocation per GPU. Using 3.98997504 GFLOP per training token from the explicit dense-matmul plus attention formula and NVIDIA's 209.5 TFLOP/s non-sparse BF16 Tensor peak with FP32 accumulation gives **76.95% approximate MFU**. The hardware denominator comes from NVIDIA's [Blackwell architecture appendix](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).

The formal gate is an absolute strict **greater than 65%** sustained median. Deterministic real-data scaling trials keep the global batch fixed at 2,097,152 prediction tokens. Four RTX 5090 GPUs measured 162,920 token/s and 77.57% median MFU; eight measured 311,806 token/s and 74.23% with the current deterministic training code. A 16-GPU BF16-gradient-reduction experiment briefly reached 65.57% with a 200 MB DDP bucket, but its 100-step gate subsequently fell to about 44% and stopped at step 15. Its continuous-versus-resumed checkpoint comparison also failed bitwise at the embedding weight. That route was rejected before formal training. The formal run therefore uses the fastest current-code configuration with robust evidence above 65%: eight GPUs. A pending allocation or short transient pass cannot displace it. Synthetic and pilot MFU do not establish benchmark quality.

## Frozen evaluation

The machine-readable formal authority is [`protocol-v2.json`](protocol-v2.json), SHA-256 `1b44371442a5c418c64f012ec5d9864718448d7bc9f81e56e7e04f34d551b4f3`. Later v3/v4 scale experiments did not pass the restored sustained-MFU and deterministic-resume gates and are not formal protocols. Protocol v2 fixes `lm-eval==0.4.9` by wheel hash; zero-shot, no-chat-template base-model evaluation; BF16, maximum length 2,048; four RNG seeds; exact task YAML hashes; immutable dataset revisions; full, unlimited evaluation; logged samples; and 10,000 within-task bootstrap replicates with seed 20260912.

The seven-task primary aggregate is the unweighted mean of HellaSwag `acc_norm`, PIQA `acc_norm`, WinoGrande `acc`, OpenBookQA `acc_norm`, ARC-Easy `acc_norm`, ARC-Challenge `acc_norm`, and BoolQ `acc`. It does not mix task sample sizes. Direct baseline claims require the same wheel, task definitions, prompts, metrics, model type, precision, context limit, and seeds.

The decontamination SQLite database contains 4,890,747 unique normalized 13-word hashes and passed SQLite integrity checking. Its SHA-256 is `661f939a8787d2c14a52b305ec6a6b5737aabe5146f557461473e89f22e3a256`. PIQA content from the evaluator's `baber/piqa` revision was verified hash-equivalent to the pinned `ybisk/piqa` copy. For BoolQ, the evaluator uses the SuperGLUE representation and decontaminates on `passage`; the pinned `google/boolq` copy contains the same passages, so every 13-word passage query is covered.

## Claim and safety boundaries

FineWeb-Edu is web data. Language and educational-score filtering does not prove factual accuracy, consent, copyright clearance, demographic balance, or absence of personal information. Exact benchmark decontamination is not a privacy filter. Tokenizer placeholder tokens and corpus metadata do not establish that PII has been removed. The resulting base model can emit false, biased, unsafe, copyrighted, or private-looking text and is not qualified for deployment.

Scheduler acceptance, GPU allocation, throughput, MFU, a decreasing pilot loss, a saved checkpoint, completed pretraining, held-out loss, frozen benchmark evaluation, and a published model are separate evidence classes. This report will only call the training complete after the optimizer reaches the frozen token target, the final checkpoint hash is recorded, held-out validation runs, export parity passes, and the seven-task evaluation finishes.
