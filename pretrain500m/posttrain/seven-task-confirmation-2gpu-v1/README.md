# Frozen two-GPU seven-task confirmation

This bundle reruns the immutable Base and evaluates the exact two F2 and two F3
long-confirmation checkpoints on two GPUs. Every model uses the same English
base-model protocol: HellaSwag, PIQA, WinoGrande, OpenBookQA, ARC-Easy,
ARC-Challenge, and BoolQ. It preserves the original zero-shot prompts, BF16
precision, context length, seeds, task versions, sample counts, and unweighted
aggregation. Rerunning Base keeps comparisons matched after the user requested
an immediate two-GPU launch.

The existing Base HF export is verified against its hash-bound export receipt.
Each native candidate checkpoint is verified by SHA-256, exported to a
job-owned temporary Hugging Face directory, checked for bitwise state parity
and bounded BF16 logit drift, and then evaluated with two RTX 5090 processes.
Temporary exports are deleted after their receipts are copied to the result
directory. The native checkpoints and all per-example evaluation logs remain
the evidence of record.

The two-GPU Base result is also compared with the earlier four-GPU Base result
to expose any execution-width effect. The resulting accuracy comparison is a
base-model capability and retention measurement. It is not an
instruction-following, safety, post-training, or market-superiority result, and
it does not promote a checkpoint automatically.
