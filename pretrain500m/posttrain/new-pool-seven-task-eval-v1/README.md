# New-pool seven-task capability screen v1

This bundle evaluates the first four completed new-pool mixture pilots with the
existing frozen two-RTX-5090 English base-model protocol. Every candidate uses
the same zero-shot HellaSwag, PIQA, WinoGrande, OpenBookQA, ARC-Easy,
ARC-Challenge, and BoolQ prompts, task versions, sample counts, seeds, BF16
precision, 2,048-token context, and unweighted aggregation.

The matched two-GPU Base aggregate is reused by exact SHA-256. Each native
candidate checkpoint is verified before export, exported into job-owned local
temporary storage, checked for bitwise state parity and bounded BF16 logit
drift, and evaluated with two physical RTX 5090 GPUs. Per-example logs and a
hash-bound result are retained; temporary Hugging Face exports are deleted.

This is an adaptive capability and retention screen because the seven-task
protocol has already been observed during model development. It does not
establish a sealed gain, instruction following, multilingual generation,
safety, formal promotion, or market superiority. N4 and the two N3 learning
rate arms will be added as a second immutable tranche after their checkpoint
identities are available.
