# Single-pass, quality-gated L20 long training

**Status at the published startup snapshot: real optimizer updates are running.**
This is a single six-hour-budget job, not the blocked multi-arm queue and not a
claim that the8.4M-image campaign has completed. The process began on2026-09-29
at11:46:51UTC on the user's existing L20. No new paid compute was provisioned.

## Frozen limits and actual exposure

| Item | This stage |
|---|---:|
| Maximum wall time |6hours|
| Maximum optimizer updates |4,096|
| Maximum distinct new TRAIN images |131,072|
| Maximum source download |48GiB|
| Source files available in the pinned plan |17PlotQA +96Docmatix shards|
| Original campaign goal, not completed |8,400,000 distinct new images|

Limits are **not completed counts or a guaranteed duration**. Source exhaustion,
user stop, the wall budget or the unchanged retention guard may end the run
sooner. A slow source must not be replaced with repeated old images to fill the
counter. Source download and compute times are recorded separately.

At the independently audited step16checkpoint, all183Adam state counters had
advanced by exactly16 from the warmstart. The7bridge and176LoRA tensors changed
and were finite. The consumed-catalog chain covers512distinct new TRAIN images.
The observed live snapshot in `startup-audit.json` is later, at32updates and
1,024new-image events; it must not be confused with the independently audited
saved checkpoint. No new full-benchmark improvement is claimed.

## Quality and retention choices

The job starts from the completed128-update C448 pilot. Its fixed teacher is
not that newer model: the teacher is the healthier pre-pilot step4608model.
The original language and vision backbones remain frozen; the bridge and
attention LoRA continue training. Input arithmetic remains unchanged T448.
No unqualified TF32/BF16/compile or H896 option was enabled.

Compared with the short pilot, bridge and LoRA peak learning rates are reduced
to3e-6. New-task CE weight is0.30, old natural-QA CE is0.40, old domain replay
CE and caption CE are0.15each. QA teacher-KL weight is1.0; caption KL is0.5.
The purpose is to reduce further old-task loss, **not a demonstrated optimal
recipe**. The warmstart old-QA score is52.8125%; the absolute original anchor
remains56.25% and the floor remains51.25%, with the existing caption/text
margins. Two consecutive failing retention checks stop the run.

Each update contains24new PlotQA images,8new single-page Docmatix images,
16old natural-QA examples,8old balanced four-domain examples and8old captions.
Text replay runs every fourth update. Old-task replay intentionally repeats
its fixed128-step pool; it is not counted as new unique images.

## Source defects found before starting

The first source review found a PlotQA absolute value labelled `3.38e+1.` even
though its source chart shows an order of magnitude near3.38e10. This is not
just trailing punctuation. Rather than guess replacement answers, this stage
excludes scientific-notation PlotQA answer candidates pending original-source
revalidation. Some valid candidates may therefore also be omitted; the rule
changes coverage and is not evidence that every scientific-notation label is
wrong. Other admissible questions on the same image can still be selected.

The review also found a low-value document containing generic book-download
promotion. Additional document-level boilerplate filters were applied before
QA selection. Original source bytes, the first preflight and the prior stream
code are preserved. `stream_sources-v1.py` is archival, not the executed code.
No benchmark answer or old heldout record was rewritten to improve a score.

The revised stream admits one complete question/answer per distinct image,
rejects multipage rows instead of attaching document-level answers to an
arbitrary page, and keeps complete targets within512total/128answer tokens.
It checks source SHA256, earlier raw/pixel/dHash identities and known document
keys. The prior exclusion cache holds585,712raw/pixel hash keys, **not that
many independently counted images**. Semantic contamination and all annotation
quality are not exhaustively certified.

## Bounded memory and checkpointing

Only two source lanes are active. Whole public source shards are verified
before consumption; the first two are kept in a bounded2GiB seed cache, and
later raw sources are held in memory rather than accumulated on disk. The
producer queues at most two prepared update batches. No per-chunk persistent
DataLoader workers or multi-arm job scheduler are used.

Checkpoints are atomic and include model increments, optimizer state, Python/
NumPy/Torch/CUDA RNG state, elapsed wall budget and the last consumed data
catalog. Catalogs produced ahead of training are not credited as trained.
The first saved update reloaded identically in the same process. A separate
CPU source-cursor test reproduced the next32real image/question rows exactly.
This is not a claim of a completed fresh-process GPU next-update equivalence
test. No old checkpoint or dataset was deleted.

## Evidence and reproduction boundary

-`startup-audit.json`: independent saved-state and catalog-chain audit.
-`first16-updates.jsonl`: actual finite-loss optimizer updates.
-`config.json`, `protocol.json`: fixed source, model, budget and loss settings.
-`qualification.json`, `cpu-tests.json`:30CPU contracts plus real-source checks.
-`quality-screening.json`: the pre-training annotation concern and exclusions.

The source files are the executed versions, but require the user's existing
hash-bound local project, checkpoints and datasets. This is not a self-contained
model release. Raw images, optimizer weights and checkpoint tensors are not in
Git. A full publication-quality before/after benchmark is still pending.

Primary references: [SmolVLM smaller models](https://huggingface.co/blog/smolervlm)
for balanced multimodal supervision; [PlotQA](https://github.com/NiteshMethani/PlotQA)
for source-data provenance and numeric reasoning tasks;
[Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix) for the
single/multipage document schema; [PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
for preserving optimizer state. These motivate the design, not a score claim.

## Later live check

At 2026-09-29T11:55:11.330965+00:00, the same long-training process was alive. See
[the timestamped snapshot](progress-snapshot-1154.json). At step64, old-QA was
53.671875%, versus52.8125% at this run's start, and all existing QA/caption/text
retention checks passed. This small diagnostic is not a public-benchmark gain.

The original review receipt's wording about two full-image inspections was too
broad. [This clarification](review-scope-clarification.json) records the actual
prelaunch review scope without altering source data, the frozen training code
or thresholds. Exhaustive semantic review of all future data is not claimed.
