# Matched external closed-book SciQ baselines v1

This evaluation scores four public base language models on exactly the same 489
development rows, frozen choice order, prompt, and likelihood metrics as the
529M Base/F2 independent knowledge proxy. The SciQ `support` field is omitted.

Each model and tokenizer snapshot is pinned to a Hugging Face commit and checked
against a local per-file SHA-256 manifest before GPU loading. OpenELM uses the
officially documented Llama 2 tokenizer, pinned separately. All model loading is
offline and uses BF16 eager attention with deterministic PyTorch algorithms.

The primary metric is `acc_norm`: accuracy after choosing the answer with the
highest mean conditional log likelihood over its answer tokens. `acc` based on
total answer-token log likelihood is secondary. Tokenization necessarily differs
between model families, but prompt text, completion text, rows, choices, scoring
formula, dtype, batch size, and tie breaking are fixed.

These are development results on a public benchmark. They provide a matched
comparison for this one closed-book English science proxy, not broad model
superiority or a sealed test result. The reserved 490 confirmation rows remain
unscored.
