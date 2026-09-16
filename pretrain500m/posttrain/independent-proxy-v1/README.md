# Independent proxy v1: frozen Belebele English tranche

This is the first independent capability proxy added after the reused seven-task
suite had already informed mixture selection. It uses the hash-pinned English
Belebele file that was reserved before this protocol was written. The 900 rows
are split, without looking at model outputs, into 450 development rows and 450
confirmation rows by a frozen SHA-256 ordering in `splits.json`.

`evaluate.py` scores the four answer strings by conditional causal-language-model
likelihood. The primary metric is answer-token mean log likelihood accuracy
(`acc_norm`); unnormalized total log likelihood accuracy (`acc`) is secondary.
The prompt, checkpoint identities, precision, tokenizer, split membership and
bootstrap seed are fixed in `plan.json`. The evaluator loads native checkpoints
directly and does not rely on an unpinned remote dataset or an HF export.

The development half may guide fresh-corpus pilot selection. The confirmation
half must remain unscored until a candidate is frozen. This tranche measures
English reading comprehension only. It is not a sealed benchmark, a complete
proxy for math/code/knowledge, or evidence of market superiority. The public
benchmark may have appeared in a model's ancestors; exact-span reservation scans
bound known project intake, but cannot prove semantic non-contamination.

