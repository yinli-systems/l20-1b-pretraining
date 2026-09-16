# N4 seven-task capability screen v2

This immutable tranche evaluates the completed N4 multilingual-mixture pilot
immediately, without waiting for the two N3 learning-rate arms. It uses the
same frozen two-RTX-5090 zero-shot HellaSwag, PIQA, WinoGrande, OpenBookQA,
ARC-Easy, ARC-Challenge, and BoolQ protocol as Base and tranche v1.

The checkpoint, matched Base aggregate, protocol, and offline cache are bound
by SHA-256. The native checkpoint is exported into job-owned local temporary
storage, checked for bitwise state parity and bounded BF16 logit drift, and the
temporary export is deleted after evaluation.

This is an adaptive capability and retention screen. It does not establish a
sealed gain, instruction following, multilingual generation, safety, formal
promotion, or market superiority.
