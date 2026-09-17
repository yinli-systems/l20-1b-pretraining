# Massive pretraining data expansion, 2026-09-17

## Decision

The current 25,000-image Vision corpus supports bounded systems and
representation experiments only. It does not support large-scale visual
pretraining. The current MoE production writer has also admitted zero tokens,
so neither its 150B qualification slice nor the planned 5.1T programme may be
described as data-ready.

Two resumable, fail-closed acquisitions are introduced:

1. Vision stage 1 downloads the 16 official CVDF Open Images train archives,
   reported as 1,743,042 images and approximately 513 GB.
2. MoE stages the official Nemotron-CC inventory, reported as 31,279 zstd
   JSONL objects, 10.4 TiB compressed, and 6.3T published tokens.
3. Vision stage 2 pins DataComp-medium's 253 parquet metadata files,
   30,638,846,406 exact bytes, and a published 128M-sample candidate pool.

Each route runs a single-object/archive smoke before its dependent full job.
Downloads are source staging, not training admission.

## Primary-source scale anchors

- The official DINOv2 model card reports that its 1.1B-parameter ViT-g/14 was
  trained on 142M images. This is the scale anchor for the Vision programme,
  not a claim that Open Images stage 1 matches DINOv2's data.
- The Open Images download page and CVDF repository report the 1,743,042-image,
  approximately 513 GB train archive subset. It is a practical first rung with
  official metadata, but it is about 81 times smaller than 142M images.
- The official DataComp repository publishes 12.8M/450 GB, 128M/4.5 TB,
  1.28B/45 TB and 12.8B/450 TB pools. DataComp small and medium are the next
  visual candidates only after metadata, rights, content, deduplication and
  contamination gates are implemented.
- The official Nemotron-CC index reports 6.3T tokens: 4.4T globally
  deduplicated real tokens and 1.9T synthetic tokens, in 10.4 TiB. Its paper
  reports that data curation and classifier-based quality partitions matter;
  the aggregate published count is not a model-specific admitted count.
- DCLM reports a 240T-token standardized corpus and a 7B baseline trained on
  2.6T tokens. Its ablations identify model-based filtering as a major source
  of downstream improvement, supporting a quality-stratified MoE mixture
  rather than indiscriminate token accumulation.

Primary sources:

- DINOv2 repository and model card: <https://github.com/facebookresearch/dinov2>
- Open Images V7 downloads: <https://storage.googleapis.com/openimages/web/download_v7.html>
- CVDF Open Images repository: <https://github.com/cvdfoundation/open-images-dataset>
- DataComp repository: <https://github.com/mlfoundations/datacomp>
- Nemotron-CC release index: <https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/index.html>
- Nemotron-CC paper: <https://arxiv.org/abs/2412.02595>
- DCLM paper: <https://arxiv.org/abs/2406.11794>

## Admission and execution gates

Vision admission requires archive and metadata identities, a per-image license
and attribution join, deterministic decoding and minimum-resolution checks,
exact and perceptual global deduplication, downstream train/validation/test
overlap removal, visual and safety audits, packed-shard identities, and an
exact admitted-image count. The training launcher must reject any receipt that
does not say `TRAINING_ADMITTED`.

MoE admission requires object identities, provenance and license review,
schema validation, quality-partition sampling, secrets/PII and unsafe-content
handling, URL/exact/near/repeated-line cross-source deduplication, exact and
fuzzy benchmark-contamination checks, frozen-tokenizer counts, sealed splits,
and human audit. Synthetic and duplicated tokens are tracked separately and
cannot silently inflate the 5.1T budget.

The completed v8 one-pipeline benchmark processed about 64.15M token IDs in
242 seconds, approximately 265k token IDs/s. At that rate, 5.1T token IDs would
take roughly 223 days before full global deduplication and audits. The next
builder therefore has to partition the 31,279-object inventory across workers,
write mergeable identity and deduplication state, and issue one deterministic
global receipt. No formal MoE training starts from raw downloaded files.

At deployment time the scheduler reported no public free RTX 4090 or RTX 5090
GPUs. `/ssd` had about 88 TB available and `/data` about 192 TB. Acquisition
jobs request one RTX 4090 each only because the cluster has no accessible CPU
partition; their smoke/full dependencies start automatically when allocated.

## Current execution

| Route | Smoke | Dependent full | Immutable source | Source-manifest SHA-256 |
| --- | ---: | ---: | --- | --- |
| Open Images 1.743M archives | `1599005` | `1599006` | `/ssd/scxi253/data-scale-source-20260917-v4` | `ca6fdac0b3253375ad0104228dffc54bf2b89b396f910ae429019bc957de301e` |
| Nemotron-CC 31,279 objects | `1598652` | `1598653` | `/ssd/scxi253/data-scale-source-20260917-v1` | `de4f5b0412bf4354f10f6b75809fc2bb5f564f28726bcc2fc72f4924ccb114d3` |
| DataComp-medium metadata (capacity blocked) | `1598945` failed | `1598946` cancelled | `/ssd/scxi253/data-scale-source-20260917-v2` | `26dff08a9b370ddd8ad7c527cb4a5ab58e05653e2b44895d549ef9864c94a260` |

Open Images `1598650` was cancelled after a verified compute-node egress
failure left its partial at zero bytes; `1598651` never allocated. The sole
corrected pair uses a Slurm-controlled login-node downloader with resumable
64 MiB segments and exact byte-length, gzip and full-file SHA-256 gates. Smoke
`1599005` is allocated and has finite segment-byte progress; full `1599006`
remains dependency-gated. DataComp `1598663` failed before download on an
unwritable target and `1598664` never allocated. Its sole corrected smoke
`1598945` then failed before download because the frozen 10 TiB free-space
floor exceeded every writable scxi253 path; `1598946` was cancelled without
allocation. No metadata or images were staged and no additional retry is
permitted until capacity is provided. Allocation, transfer, completion and
admission remain separate states.
