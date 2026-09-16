# Vision 1B pretraining qualification

This directory contains a bounded systems qualification for a DINO-style
ViT-g/14 vision backbone. It is not a trained image model and does not admit a
dataset to training.

The first gate uses deterministic synthetic tensors to answer only whether an
approximately 1.1B-parameter student plus an equally sized frozen teacher can
complete forward, backward, and optimizer steps on one eight-GPU RTX 4090
node. It exercises the parts that dominate a self-supervised vision run:

- ViT-g/14 geometry: width 1536, depth 40, 24 heads and SwiGLU feed-forwards;
- two 224-pixel global views and eight 98-pixel local views per source image;
- BF16 compute with FP32 parameters, reductions, and AdamW state;
- composable FSDP2 sharding and per-block activation checkpointing;
- optional `torch.compile` using `max-autotune-no-cudagraphs`.

Synthetic throughput is an engineering measurement. It excludes image decode,
augmentation, storage and data-loader costs and cannot support an accuracy,
quality, convergence, or pretraining-completion claim.

Formal training remains blocked until an immutable image corpus passes source,
license, attribution, duplicate/benchmark-overlap, integrity, quality and split
audits. ImageNet also requires authorized access under its terms.

## Qualified runtime recipe

The 2026-09-17 synthetic qualification selected a four-GPU FSDP2 group with
batch 64 per rank, fused global/local crop execution, BF16 all-gathers, FP32
parameter shards and optimizer state, activation checkpointing, persistent
student and teacher parameters between forward and backward, Ring/Simple NCCL,
and `max-autotune-no-cudagraphs`. Slurm job `1595805` completed 20 steps with
16 measured steps at a median 50.1237% useful MFU and 64.1728% HFU. It processed
44.0108 source images/s and reserved 21.6582 GiB per GPU at peak.

Eight GPUs should use two replicated four-way shard groups (HSDP), rather than
one eight-way FSDP group. Job `1595813` completed with 81.6269 source images/s,
46.4823% useful MFU and 92.7352% throughput scaling efficiency relative to two
four-GPU groups. Replica-gradient synchronization accounts for the lower
per-GPU MFU and should be amortized with gradient accumulation in a formal run.

The evidence receipt is in
`reports/vitg14-1b-synthetic-preflight-20260917.json`. These numbers describe the
synthetic proxy only. The proxy uses normalized-projection MSE and omits the
full DINO/iBOT objectives, teacher EMA update, image decode and augmentation,
checkpoint I/O and downstream evaluation.

The target follows the official DINOv2 ViT-g/14 scale: the upstream model card
describes a 1.1B-parameter backbone trained on 142M images. A faithful project
must pin the upstream method and independently admit a suitable image corpus
before these runtime settings become a formal pretraining recipe.

- DINOv2 repository: <https://github.com/facebookresearch/dinov2>
- DINOv2 model card: <https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md>
- ImageNet access terms: <https://www.image-net.org/download.php>
