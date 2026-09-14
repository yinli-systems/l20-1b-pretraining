# Frozen seven-task confirmation

This bundle evaluates the exact two F2 and two F3 long-confirmation
checkpoints with the same English base-model protocol used for the immutable
parent: HellaSwag, PIQA, WinoGrande, OpenBookQA, ARC-Easy, ARC-Challenge, and
BoolQ. It preserves the original zero-shot prompts, BF16 precision, context
length, seeds, task versions, sample counts, and unweighted aggregation.

Each native model-only checkpoint is verified by SHA-256, exported to a
job-owned temporary Hugging Face directory, checked for bitwise state parity
and bounded BF16 logit drift, and then evaluated with four RTX 5090 processes.
Temporary exports are deleted after their receipts are copied to the result
directory. The native checkpoints and all per-example evaluation logs remain
the evidence of record.

The resulting accuracy comparison is a base-model capability and retention
measurement. It is not an instruction-following, safety, post-training, or
market-superiority result, and it does not promote a checkpoint automatically.

