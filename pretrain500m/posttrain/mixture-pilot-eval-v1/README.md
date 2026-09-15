# Mixture-pilot capability screen v1

This bundle evaluates the three development-shortlisted current-corpus pilots
against the already completed, hash-bound two-RTX-5090 Base result. Each Slurm
job evaluates one candidate on the same frozen seven English base-model tasks:
HellaSwag, PIQA, WinoGrande, OpenBookQA, ARC-Easy, ARC-Challenge, and BoolQ.

The three candidates cover the best overall development loss (`r3-lr1e4`), the
best common-learning-rate control (`r0-lr6e5`), and the code-guard mixture that
was effectively tied with that control (`r4-lr6e5`). Running one candidate per
two-GPU job permits concurrent evaluation without repeating the Base model.

Every job verifies the frozen plan, candidate checkpoint, evaluation cache,
protocol, and matched two-GPU Base aggregate before exporting the native
checkpoint. The export must pass bitwise state parity and bounded BF16 logit
drift. Temporary Hugging Face exports are stored under node-local `/tmp` and
deleted after evaluation.

`submit.py` refuses duplicate active jobs or less than 10 GiB of shared space,
then submits the three independent evaluations and records their Slurm IDs in a
single immutable receipt.

This is an adaptive capability and retention screen over a previously used
seven-task protocol. It does not constitute sealed evaluation, instruction
following, multilingual generation, safety, or market-superiority evidence,
and it cannot promote a checkpoint automatically.
