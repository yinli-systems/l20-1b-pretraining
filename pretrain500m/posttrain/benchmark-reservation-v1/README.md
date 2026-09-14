# Supplemental benchmark reservation

This bundle reserves reference content for contamination checking. None of these
records is training data. Acquisition does not implement a contamination matcher,
scan any training corpus, score a model or establish a capability improvement.
The existing seven-task benchmark protection remains separately required.

All 32 downloaded files match their pinned full LFS SHA256 or Git blob identity
and expected byte size. The 12,016,431 source bytes produce 11,830 records and
53,073 separately labeled fields. Raw documents and provenance are retained;
fields are not concatenated into artificial cross-field n-grams. Each original
row, field and source file has a content digest. The model never receives this
reference bundle as training input.

| Reference | Reserved content | Origin |
|---|---|---|
| HumanEval | 164 problems, prompts, reference solutions and tests | [OpenAI dataset](https://huggingface.co/datasets/openai/openai_humaneval), [original release](https://github.com/openai/human-eval) |
| MBPP | 974 full and 427 sanitized records across all supplied splits; variants remain identifiable | [Google Research dataset](https://huggingface.co/datasets/google-research-datasets/mbpp) |
| IFEval | 541 instruction prompts; original instruction metadata stays in raw files | [Google dataset](https://huggingface.co/datasets/google/IFEval) |
| Belebele | 900 records each for Arabic, German, English, French, Japanese, Spanish and simplified Chinese | [Meta dataset](https://huggingface.co/datasets/facebook/belebele) |
| MGSM | All eleven original 250-question TSVs and 174 long string literals from the exemplar source | [Google Research original release](https://github.com/google-research/url-nlp/tree/452a21ad3dae5668c06ceeac21ff073e1e40f9be/mgsm) |
| MATH-500 | 500 problems, solutions and answers; this is a subset, not all of MATH | [Maintained subset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) |

MGSM is fetched from the original repository. Python exemplar files are parsed
with AST only; imports and executable code are never run. MBPP full/sanitized
variants and translated questions are overlapping references, so these record
counts are not counts of independent evaluation problems. Pinned MATH-500
metadata declares no dataset license; it remains a reference-only reservation,
and no training-license claim is made. Other declared licenses and immutable
revisions are retained in `inventory.json` and actual content receipts.

## Remaining matcher work

The earlier `[a-z0-9]+` 13-word matcher does not provide the needed Unicode and
code coverage. A replacement must freeze its normalization and overlap rules,
preserve operators, identifiers and numbers, handle Chinese/Japanese without
assuming space-separated words, and report covered and skipped fields. Common
short choices and numeric answers must not independently cause arbitrary training
documents to be removed. Whitespace/case changes, punctuation and code/math
meaning changes require explicit positive and negative controls. Do not describe
an exact-span audit as semantic or translation-level decontamination. Record
each exclusion's source row and benchmark reference identity.

[DataTrove](https://github.com/huggingface/datatrove) provides separate n-gram
decontamination and multilingual processing components; its availability does not
by itself qualify this project's normalization or corpus coverage. The actual
matcher and complete combined-corpus scan are still pending.

After integrity verification, keep this bundle under the remote evaluation
directory, outside all candidate training pools. It can supply reservation
references for a future frozen matcher. Generated benchmark-code execution and
model scoring require their separate evaluation workflow.
