# External maximum-throughput/MFU review context

## Objective

Review this repository and the attached original research programme as a
systems-and-ML research expert.  Find the highest defensible **measured**
training token throughput and MFU attainable on ParaCloud while preserving the
specified model arithmetic, frozen-data protocol, reproducibility, and quality
gates.  Do not optimize a diagnostic shortcut and call it valid training.

The requested end state is a genuinely novel 7.002B-total / 1.189B-active
Top-2 MoE.  The current candidate is intentionally stopped at a 277.8M-total /
79.6M-active oracle proxy because its scientific gates failed.  A systems
redesign may continue at proxy scale; 7B training must remain gated on quality.

## ParaCloud environment

- Login/storage: `ln01`, `/ssd/scxi253/cvcr-research`; home quota is too small.
- Training partition actually validated: `gpu_4090`, one node, 4 x RTX 4090.
- Slurm shape: `--nodes=1 --gres=gpu:4`; partition rejects explicit memory
  flags, so scripts omit `--mem*`.
- Runtime: Python 3.12.13, PyTorch 2.11.0+cu128, CUDA runtime 12.8,
  cuDNN 9.19, NCCL 2.28.9, Triton 3.6.0, NumPy 2.3.5.
- Compilation/cache: `torch.compile(mode="default")`; Triton, Inductor and
  Python caches use exact job-local `/tmp` directories.
- Arithmetic: BF16 attention/expert compute; FP32 router softmax/reductions;
  native `torch.nn.functional.grouped_mm`; PyTorch Flash SDPA with GQA.
- Frozen packed data: 2,049-token records, model sequence length 2,048;
  revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, status
  `FROZEN_VERIFIED_PACK`.
- Public-capacity snapshot at review preparation: 47 RTX 4090 cards free and
  zero public RTX 5090 cards free.  Capacity is time-specific and not a
  performance result.

Selected installed versions:

```text
torch==2.11.0+cu128
triton==3.6.0
numpy==2.3.5
nvidia-cublas-cu12==12.8.4.1
nvidia-cudnn-cu12==9.19.0.56
nvidia-nccl-cu12==2.28.9
nvidia-cusparselt-cu12==0.7.1
transformers==5.15.1
tokenizers==0.22.2
safetensors==0.8.0
```

## Current implementation

- 8-layer oracle proxy: hidden 768, SwiGLU expert width 768, 16 experts,
  dropless Top-2, 12 Q heads / 4 KV heads, sequence 2,048.
- Exact requested design config: 24 layers, hidden 2,048, expert width 2,816,
  16 experts, Top-2, 32 Q / 4 KV heads, untied 32K embeddings.
- DDP across four GPUs; each rank holds the complete proxy model.
- AdamW fused optimizer; microbatch 8/GPU; gradient accumulation 8.
- Expert dispatch sorts token assignments and invokes jagged grouped GEMM.
- CVCR probes are detached from expert weights and run once every four
  accumulation microbatches with four times the conditional budget.
- Predictor/probe modules are removed from the inference state.
- Local regression suite: 14 passing tests.

Read all source files, tests, configs, Slurm scripts and reports in the archive.
In particular inspect `cvcr_moe/model.py`, `cvcr_moe/cvcr.py`,
`cvcr_moe/train.py`, `scripts/profile_methods.py`, and all run manifests and
metrics included under `evidence/`.

## Measured systems evidence

Fastest short compiled screen on 4 x RTX 4090:

| Method | tok/s | Active MFU | Hardware-work MFU | Overhead |
|---|---:|---:|---:|---:|
| Top-2 | 387,867.89 | 29.10% | 29.10% | baseline |
| Rank-8 CVCR interval-4 | 378,696.31 | 28.41% | 28.49% | 2.42% |

Full 199,753,728-token run medians over 361 steps after 20 warm-up steps:

| Method | tok/s | Active MFU | Hardware-work MFU | Overhead vs Top-2 |
|---|---:|---:|---:|---:|
| Pure Top-2 | 379,433.93 | 28.464% | 28.464% | baseline |
| Rank-8 instrumented control, credit coefficient 0 | 370,543.24 | 27.797% | 27.875% | 2.399% |
| Rank-8 CVCR, credit coefficient 0.05 | 363,177.24 | 27.244% | 27.321% | 4.476% |

The MFU denominator is four GPUs times 209.5 dense BF16 TFLOP/s, with no
structured-sparsity multiplier.  Audit the FLOP formula and report both active
model FLOPs and actual hardware work.  Do not improve MFU merely by changing
the numerator or denominator.

Peak allocated memory was roughly 20.2 GB/GPU in the proxy runs.  First-step
compile warm-up took about 35 seconds; steady steps were around 1.38-1.42
seconds for 524,288 prediction tokens/step.

## Scientific evidence and stop state

- Rank-8 inactive-credit MSE reduction versus default: 16.63%; required 30%.
- Rank-16 predictor-only MSE reduction: 16.42%; doubling rank did not help.
- Rank-8 estimator variance reduction versus raw probing: 53.35%; versus the
  default control variate: only 10.47%.
- 64 exact downstream reroutes: local/actual regret Pearson 0.206 and Spearman
  0.199; local first-order credit is a weak downstream proxy.
- Strict rank-8 credit-only comparison (`lambda=0.05` versus identical
  instrumented `lambda=0` control): CVCR worsened CE by +0.023548 nats/token;
  paired 95% frozen-block bootstrap interval [+0.021332, +0.025851].
- Therefore the existing CVCR estimator must not be scaled to 7B unchanged.

## Required review output

1. Audit every performance-critical path and identify correctness issues,
   hidden synchronization, graph breaks, redundant casts/copies/sorts,
   load-imbalance costs, optimizer/DDP costs, and inaccurate FLOP accounting.
2. Research current primary sources and official documentation for PyTorch
   grouped GEMM/compile/SDPA, CUDA/Triton, NCCL, Megatron-Core MoE, expert
   parallelism, DeepSpeed/Tutel alternatives, fused optimizers, FP8, activation
   checkpointing, and 4 x 4090/5090 constraints.  Link sources and distinguish
   documented capability from measured evidence.
3. Produce a prioritized optimization table with exact code locations,
   expected throughput/MFU effect ranges, memory effect, correctness risk,
   implementation effort, and a microbenchmark or ablation that can falsify
   each claim.
4. Propose the best proxy execution topology and the best 7B topology.  Include
   DDP/FSDP/ZeRO, tensor/pipeline/context/expert parallel choices and explain
   why.  Account for optimizer state and total expert weights, not only active
   FLOPs.
5. Give concrete patch-level recommendations or diffs for the top changes.
   Prefer changes that can be validated independently before a long run.
6. Separate the systems path from a scientifically credible estimator
   redesign.  Suggest how to replace the weak local-credit target without
   invalidating the ordinary Top-2 inference graph.
7. End with a staged experiment matrix and hard promotion gates.  Never claim
   a projected speedup, scheduler acceptance, or profile as a measured result.

The goal is maximum real token throughput and maximum honest MFU, not a
cosmetic metric increase and not quality-invalid arithmetic.
