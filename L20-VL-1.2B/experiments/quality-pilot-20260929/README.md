# Real VLM quality training: first saved updates

This run advances beyond the earlier no-update kernel and evaluation tests.
The C448 control has performed real AdamW updates on the existing L20, saved
new checkpoint state, and passed a same-process checkpoint reload check.
This is a startup record, not a claim of final benchmark improvement.

## Frozen pilot

- Healthy parent: previous stream step4608; manifest SHA-256 `3751ac488fa45c20d29b2cf231ee9593ec59d7ee52cf24e15724e3e75489efa7`.
-128 planned updates;64 image events/update, including32 new images.
-4,096 distinct new training images:3,072 PlotQA and1,024 single-page Docmatix.
-Each step also includes16 old natural-image QA,8 old balanced four-domain QA,
 and8 old captions. A4,096-token text replay batch runs every fourth update.
-The repaired union-of-holdouts split is unchanged. The26 data-contract tests
 passed again and all4,096 selected complete targets passed token-length bounds.
-The LM and vision parents remain frozen; the bridge and attention LoRA train.
-The absolute old-QA anchor remains56.25%, with51.25% stop floor, plus the
 original caption/text margins. Two consecutive failed gates stop the run.
-No public benchmark or final confirmation is accessed to select checkpoints.

The trainer supports C448, aspect-preserving A448, and H896 with a trainable
packing projection. **Only C448 is started in this snapshot.** Submission of
the automatic queue for the other two arms was blocked by the tool safety
check and was not retried or rerouted. It did not interrupt the existing C448
process. Other arms must not be described as running or completed.

## First32-step evidence

-1,024 distinct new images used in the saved lineage;2,048 total image events.
-All32 logged updates have finite losses and gradient norms.
-First updated weights differ from the parent; checkpoint reload max parameter
 difference is0.0; the optimizer and RNG were restored. This is a same-process
 reload check, not a fresh-process next-update equivalence test.
-At32steps the old-QA probe is52.65625%, above the51.25% floor but **below
 the healthy parent's55.234375%**. Passing the gate is not zero regression.

| Fixed development diagnostic (32images/task) | Parent | Step32 |
|---|---:|---:|
| PlotQA answer+EOS NLL |3.212722715|2.190944915|
| Single-page Docmatix answer+EOS NLL |2.331584163|2.253513020|
| PlotQA normalized exact match |0/32|1/32|
| Docmatix normalized exact match |0/32|0/32|

These small diagnostic changes do not establish public ChartQA/DocVQA gains.
NLL here includes EOS and must not be compared to historical non-EOS NLL as
if the metrics were identical. Long-form Docmatix exact matching is a limited
diagnostic, not a comprehensive semantic quality metric.

## Efficiency and storage boundary

The trainer uses the bounded ordered thread prefetcher, no per-chunk persistent
DataLoader workers, no network downloads, no file deletion and no new paid
compute. Prior TF32/BF16/compiler candidates were not silently enabled. The
three-check evaluation reuse module is separate; no full-training speedup is
claimed from its earlier1.2983x evaluation result.

The8.4M distinct-image goal remains a later campaign target. This128-update
pilot is not50x data scaling. The loss log's `weighted_loss_sum` is only the sum
of its `new` and `replay` fields, not the complete weighted objective; separate
QA/caption/KL/text components remain available.

## Reproduction boundary

`train_quality.py` is the byte-identical source executed on the existing remote
project. It intentionally requires the user's hash-bound local parents, replay
files, admission manifests and images; these assets are not in this public
repository. No weights, optimizer state or source images are published here.
The original parent is not overwritten and checkpoint writes are atomic.

Primary design references: [SmolVLM](https://arxiv.org/abs/2504.05299) motivates
separate data/resolution/tokenization experiments; [PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models)
requires optimizer state when continuing training. These references motivate
the design, not a claim that this small pilot matches published model scores.
