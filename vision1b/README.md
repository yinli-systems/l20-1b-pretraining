# Vision 1B pretraining qualification

This directory contains a synthetic systems qualification and a bounded
real-image training pilot for a DINO-style ViT-g/14 vision backbone. It is not
a released or corpus-scale pretrained image model.

The first gate uses deterministic synthetic tensors to answer only whether an
approximately 1.1B-parameter student plus an equally sized frozen teacher can
complete forward, backward, and optimizer steps on one eight-GPU RTX 4090
node. It exercises the parts that dominate a self-supervised vision run:

- ViT-g/14 geometry: width 1536, depth 40, 24 heads and SwiGLU feed-forwards;
- two 224-pixel global views and eight 98-pixel local views per source image;
- BF16 compute with FP32 parameters, reductions, and AdamW state;
- composable FSDP2 sharding and per-block activation checkpointing;
- optional `torch.compile` using `max-autotune-no-cudagraphs`.
- opt-in torchao FP8 and configurable block-checkpoint frequency for paired
  screening without changing the BF16 defaults.

Synthetic throughput is an engineering measurement. It excludes image decode,
augmentation, storage and data-loader costs and cannot support an accuracy,
quality, convergence, or pretraining-completion claim.

Corpus-scale training remains blocked pending a much larger admitted corpus,
benchmark-overlap analysis, preregistered downstream evaluation, and a faithful
full DINOv2/iBOT implementation. ImageNet also requires authorized access under
its terms.

## Bounded real-image pilot

The admitted pilot uses a deterministic 25,000-image subset of the Open Images
V6 train split. The acquisition receipt freezes the official metadata file,
selection seed, exact 25,000-row manifest, per-image bytes and SHA-256 hashes,
license and attribution fields, decode/integrity checks, exact-file deduplication,
and a manual review of all 64 cells in four deterministic contact sheets. The
sample contains people and minors and includes non-explicit adult imagery, so
admission is limited to this internal representation-learning pilot. A 64-image
review cannot establish the suitability or legal status of every image.

The real objective uses a three-layer DINO projection head, row-normalized
prototypes, Sinkhorn teacher assignments, an EMA teacher, and KoLeo loss on raw
normalized backbone CLS tokens. FP16 compute uses FP32 parameters, reductions,
loss and optimizer state with static loss scaling. Every run fails closed on
non-finite gradients, low feature diversity, sample-invariant teacher logits,
near-uniform teacher assignments, median end-to-end MFU below 50%, or checkpoint
identity drift.

The 4xRTX-4090 batch-80 qualification in job `1596192` completed 25 real-image
steps without representation or teacher collapse. Loss fell from 9.25 to 8.97,
the final feature nearest-neighbor distance reached 1.07, and final teacher
entropy was 8.792 versus the 8.961 ceiling. Its median end-to-end MFU was only
49.4005%, so the efficiency gate rejected it even though the step-25 checkpoint
was saved. This negative result is retained rather than rounded up to a pass.

Two 4xRTX-5090 routes subsequently completed the full bounded 250-step run.
Batch 64 jobs `1596084` and `1596346` passed at 56.4352% median end-to-end MFU,
62.8412 source images/s and 18.77 GB peak reserved memory per GPU; final loss
was 8.9126, teacher entropy was 8.6793, and feature nearest-neighbor distance
was 1.3472. Batch 112 jobs `1596349` and `1596350` were the faster selected
route, passing at 57.6974% median end-to-end MFU, 64.2468 source images/s and
28.32 GB peak reserved memory per GPU. Its final loss was 8.9033, teacher
entropy was 8.6794, feature nearest-neighbor distance was 1.3406, and the mean
rank gradient norm was 0.2205. Both final checkpoints and their identity-bound
manifests were written at step 250.

These are training-system and objective-health results on 25,000 admitted
images. They do not establish downstream accuracy, representation quality,
corpus-scale convergence, or superiority over another model. The frozen
diagnostic below is the first downstream gate; broader natural-image transfer,
robustness, contamination and fine-tuning evaluations remain open.

## Frozen-feature downstream diagnostic

The preregistered evaluation uses the complete official 60,000-example train
and 10,000-example test splits of Fashion-MNIST and MNIST. It compares the same
architecture at deterministic random initialization, the step-250 EMA teacher,
and the step-250 student. Images are repeated to RGB, resized to 224 pixels and
ImageNet-normalized. The two classifiers are similarity-weighted 20-NN at
temperature 0.07 and a ridge probe selected on a deterministic stratified
80/20 split before refitting on the full training set.

| Frozen backbone | Fashion-MNIST 20-NN | Fashion-MNIST ridge | MNIST 20-NN | MNIST ridge | Four-score mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random initialization | 64.74% | 54.44% | 54.19% | 35.86% | 52.31% |
| EMA teacher, step 250 | 65.41% | 59.86% | 54.58% | 41.95% | 55.45% |
| Student, step 250 | 67.10% | 69.91% | 63.61% | 70.76% | 67.85% |

Job `1597849` completed the full protocol on one RTX 5090 in 24 minutes 52
seconds with exit code 0. The teacher gained 3.1425 percentage points over the
random baseline across the four primary top-1 scores, passing the frozen
2-point promotion gate. The student gained 15.5375 points over random and
12.3950 over the teacher. This makes the student the stronger representation
for the next evaluation gate; this diagnostic alone does not identify why the
bounded-run EMA teacher trails it.

The protocol, executable, raw result JSON, source receipt, GPU telemetry and
artifact hashes are in [`frozen_eval_protocol_v1.json`](frozen_eval_protocol_v1.json),
[`evaluate_frozen_features.py`](evaluate_frozen_features.py), and
[`reports/vitg14-frozen-eval-20260917/`](reports/vitg14-frozen-eval-20260917/).
Both datasets are low-resolution grayscale controls. These results cannot
establish ImageNet performance, broad visual quality, release readiness or
superiority over another model.

## Qualified runtime recipe

The fastest measured four-GPU recipe now uses RTX 5090, four-way FSDP2, BF16,
batch 112 per rank, full per-block activation checkpointing, persistent student
and teacher parameters between forward and backward, fused multicrop execution,
Ring/Simple NCCL and `max-autotune-no-cudagraphs`. Job `1595879` completed at
66.3996 source images/s, 59.6315% useful MFU and 76.3455% HFU. It reserved
30.3086 GiB per GPU and was 50.87% faster than the selected four-RTX-4090 job.

For a real input pipeline, job `1595846` is the recommended headroom variant:
it reshards the student after forward, reserves 28.3926 GiB and retains 99.81%
of the maximum synthetic throughput. This margin is more valuable than the
0.19% synthetic gain once decode, augmentation and the full objective are
present.

The paired frontier rejected several plausible shortcuts on this exact stack:

- batch 116 and 128 were slower than batch 112;
- removing checkpointing at batches 28, 30 and 32 exhausted the 32GB cards;
- checkpointing every second block at batches 40 and 44 also exhausted memory;
- tensorwise FP8 plus FP8 FSDP all-gather ran correctly but was 37.8% slower
  than BF16 batch 112;
- xFormers 0.0.35 was numerically compatible on the tested attention shape but
  its forward/backward path was 31.8% slower than native PyTorch SDPA.

The complete 5090 receipt, source hashes and negative results are in
`reports/vitg14-1b-extreme-optimization-20260917.json`. FP8 remains an opt-in
research control because official torchao results are hardware- and
shape-dependent; it is not the selected recipe for these RTX 5090 jobs.

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
- Open Images facts and figures: <https://storage.googleapis.com/openimages/web/factsfigures.html>
- ImageNet access terms: <https://www.image-net.org/download.php>
- torchao quantized training: <https://docs.pytorch.org/ao/stable/workflows/training.html>
- PyTorch float8 and FSDP2 study: <https://pytorch.org/blog/training-using-float8-fsdp2/>
- PyTorch compile throughput study: <https://pytorch.org/blog/maximizing-training-throughput/>
- xFormers optimized operators: <https://facebookresearch.github.io/xformers/components/ops.html>
