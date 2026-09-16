# Independent code proxy v1: MBPP sanitized execution

This is the code component of the new development proxy. It uses the pinned
MBPP sanitized test parquet already held outside training. An output-blind
SHA-256 ordering uses task ID and prompt only to split 257 tasks into 128
development and 129 confirmation tasks. Only development may be scored during
mixture work.

The primary metric is greedy execution pass@1 against every provided unit test.
Generated code runs with an AST policy, isolated Python, a fresh user and
network namespace, resource limits, no stdin, and a three-second wall timeout.
Syntax/policy pass rate and gold-code token NLL are secondary diagnostics. They
do not substitute for execution correctness.

MBPP is public and may occur in model ancestors. The project exact-span scans
reduce known local intake leakage but cannot prove semantic non-contamination.
This development proxy is not sealed and cannot establish broad software
engineering ability or model-market superiority.
