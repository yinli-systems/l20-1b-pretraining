# 529M pretraining and data-mixture experiments

This directory contains the reproducible source, frozen protocols, tests, and
compact receipts for an isolated 528,748,800-parameter language-model
experiment. Large corpora, model checkpoints, evaluation caches, and vendored
binary packages are intentionally excluded from Git.

The immutable parent checkpoint is step 7,629 after 15,999,172,608 prediction
tokens. Its SHA-256 is
`13aa21721e15c48cdfdafe30d8fdd41d9c95c90be766327661d96af1e90dd6cf`.

The completed work includes:

- provenance-bound intake, exact deduplication, contamination screening,
  family-aware splitting, quality filtering, and train/development/confirmation
  packing;
- deterministic four-GPU restart qualification, plus 8- and 16-GPU scaling
  qualifications;
- a frozen four-recipe, two-seed screen;
- four long confirmation runs comparing the F2 reasoning and F3 broad
  multilingual recipes, each trained for 2,147,483,648 prediction tokens on
  four RTX 5090 GPUs.

See [RESULTS.md](RESULTS.md) for the result table and current claim boundary.
The post-training machinery and stage definitions are under
[posttrain/](posttrain/).

## Reproducibility boundary

Receipts bind source revisions, manifests, protocol hashes, Slurm job IDs,
hardware identities, checkpoint digests, and measured metrics. The repository
does not contain the large model or corpus payloads referenced by those
receipts. A completed training job is evidence of training and artifact
integrity; it is not by itself evidence of downstream task accuracy or model
superiority.
