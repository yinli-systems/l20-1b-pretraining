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

## Bounded step-250 to step-1000 continuation

Job `1598538` allocated four RTX 5090 GPUs on node `wqd10nah08g5` and strictly
loaded the selected batch-112 checkpoint. Steps 251 through 259 were finite;
steps 252 through 259 had 56.41% median end-to-end MFU and telemetry retained
at least 4,972 MiB of free device memory. Rank 3 then lost its CUDA device and
terminated the NCCL watchdog before the first new checkpoint. Invalid Triton
autotune candidates appeared earlier in the log, but the later evidence below
means they cannot be assigned as the root cause.

The sole permitted replacement, job `1598829`, kept every training argument
unchanged, used fresh local Inductor/Triton caches, and isolated autotune
candidates in subprocesses. Slurm placed it on the same node. It failed in the
initial `nvidia-smi` identity gate with `Unable to determine the device handle
for GPU3 ... Unknown Error`, before Python training, metrics or checkpoints.
This corroborates a node/GPU3 infrastructure fault and rules out the wrapper
change as a recovery. The retry cap is now exhausted; another submission must
be explicitly authorized and must exclude `wqd10nah08g5`.

The parent step-250 checkpoint was revalidated before the replacement: it
contains six files and 18,354,419,036 bytes, frozen by a receipt whose SHA-256
is `c1e71d6c98091806ea35d1edba74fad256f828c7ff306d0d58722d21dcaaa0f1`.

The continuation does not silently extend the original 250-step cosine
horizon. It uses an explicit second-stage schedule: learning rate moves
continuously from the parent's terminal `1e-5` to `1e-6`, EMA momentum moves
from `0.9999` to `1.0`, and checkpoints are written at steps 500, 750 and
1,000. The fastest qualified real-image layout remains fixed at four RTX 5090
GPUs and batch 112 per rank. The job fails closed below 50% median end-to-end
MFU, with less than 2 GiB of GPU memory headroom, on checkpoint drift, or on
any inherited finite-value, gradient, feature-diversity or teacher-collapse
gate.

This adds 336,000 deterministic image exposures from the same admitted 25,000
files. It is a bounded overtraining test, not new data or corpus-scale
pretraining. The identical frozen diagnostic is preregistered for steps 500
and 1,000. The continuation is promoted only if step 1,000 beats the step-250
student's 67.8450% four-score mean without a material per-dataset regression;
otherwise step 250 remains selected.

The machine-readable contract is
[`continuation_25k_stage2_protocol_v1.json`](continuation_25k_stage2_protocol_v1.json).
The original immutable source remains
`/ssd/scxi253/vision1b-research/source-cont-20260917-v1`. The replacement uses
the read-only source
`/ssd/scxi253/vision1b-research/source-cont-20260917-v2`, whose
`SOURCE_SHA256SUMS` SHA-256 is
`65fa783ebe9e63324f48d921c8e720fd7a9625e2768728ff45b28f81c62c9b4e`.
The failure and recovery evidence is frozen in
[`reports/continuation-recovery-20260917/receipt.json`](reports/continuation-recovery-20260917/receipt.json).

## Corpus-scale expansion

The 25,000-image continuation is deliberately bounded. It cannot support a
claim of large-scale visual pretraining. The first corpus-scale acquisition
stage therefore targets the official CVDF Open Images train archives: 16
archives, 1,743,042 reported images, and approximately 513 GB. A one-archive
smoke job must complete before its dependent full download can run.

Downloaded archives remain `NOT_ADMITTED`. Training admission additionally
requires an identity-bound metadata join, per-image license and attribution,
successful RGB decoding and dimension checks, exact and perceptual
deduplication, downstream split-overlap removal, deterministic content and
safety audits, and a sealed receipt. The next scale rung is a separately
filtered DataComp pool; its published small and medium pools contain 12.8M and
128M samples. Neither published counts nor downloaded bytes are accepted as
training examples.

The frozen acquisition contract and resumable implementation are
[`openimages_cvdf_1p7m_protocol_v1.json`](openimages_cvdf_1p7m_protocol_v1.json)
and [`acquire_openimages_cvdf_1p7m.py`](acquire_openimages_cvdf_1p7m.py).
Research, scale choices, and promotion boundaries are recorded in
[`../reports/massive_pretraining_data_expansion_20260917.md`](../reports/massive_pretraining_data_expansion_20260917.md).

Open Images jobs `1598650`/`1598651` are the smoke/full pair. They use immutable
source `/ssd/scxi253/data-scale-source-20260917-v1`, whose source-manifest
SHA-256 is `de4f5b0412bf4354f10f6b75809fc2bb5f564f28726bcc2fc72f4924ccb114d3`.
DataComp-medium metadata jobs `1598663`/`1598664` are a second smoke/full pair.
They pin all 253 parquet files and 30,638,846,406 bytes at dataset revision
`8af865e284668a1c52d12846eff0a9d6f1da6ec6`. Their immutable source-manifest
SHA-256 is `26dff08a9b370ddd8ad7c527cb4a5ab58e05653e2b44895d549ef9864c94a260`.
Both smoke jobs were scheduler-accepted and were pending priority at submission;
their dependent full jobs cannot start early.

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

## Frozen reference comparison

Six official pretrained references completed the same full frozen-feature
protocol with exit code 0 and verified artifact hashes.  Every row below uses
the same Fashion-MNIST and MNIST files, 224-pixel ImageNet-normalized
preprocessing, weighted 20-NN and ridge-probe implementation.  The four-score
mean is the unweighted mean of the four displayed top-1 scores.

| Frozen backbone | Fashion-MNIST 20-NN | Fashion-MNIST ridge | MNIST 20-NN | MNIST ridge | Four-score mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vision 1B student, 25k images / step 250 | 67.10% | 69.91% | 63.61% | 70.76% | 67.85% |
| DINOv2-g/14 | 91.57% | 91.95% | 92.39% | 96.37% | **93.07%** |
| DINOv2-B/14 | 90.22% | 89.92% | 94.72% | 96.85% | 92.93% |
| DINOv2-L/14 | 90.61% | 90.66% | 91.50% | 95.12% | 91.97% |
| DINOv2-S/14 | 89.54% | 88.79% | 94.09% | 95.38% | 91.95% |
| Supervised ImageNet-1K ResNet-50 | 88.47% | 87.48% | 95.10% | 96.13% | 91.80% |
| OpenAI CLIP ViT-B/32, strict shared preprocessing | 87.14% | 85.14% | 96.44% | 95.57% | 91.07% |

The bounded Vision 1B student trails the lowest reference by 23.2275
percentage points and DINOv2-g/14 by 25.2250 points.  It therefore passed its
preregistered random/teacher promotion gate and learned transferable features,
but it does not approach these mature pretrained representations under this
diagnostic.  DINOv2, CLIP and supervised ResNet use different and vastly larger
training regimes, so the table does not compare data efficiency, compute
efficiency, native preprocessing or overall model quality.

The OpenCLIP runtime warns at empty model construction that pretrained weights
were not loaded by `create_model`; the evaluator then verifies the official JIT
file's exact byte length and SHA-256, removes only its three metadata buffers,
and calls `load_state_dict(..., strict=True)` before evaluation.  This warning
does not indicate a random-weight result.

Jobs `1598045`, `1598047`, `1598049`, `1598051`, `1598088` and `1598116`
produced the full reference rows.  Exact result JSON, source receipts, GPU
telemetry, remote manifests and the computed comparison receipt are preserved
in [`reports/reference-baselines-20260917/`](reports/reference-baselines-20260917/).

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
