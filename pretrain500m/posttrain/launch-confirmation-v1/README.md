# Immediate long-confirmation launcher

The background launcher waits for the hash-bound filter/pack completion receipt,
re-runs all 38 CPT unit tests on the remote runtime, builds and admits exact F2
and F3 manifests, checks the four-output storage floor, and submits both recipes
at both seeds. The four 4-GPU jobs request 16 GPUs from the account-authorized
`gpu_5090` partition.

Submission is not reported as training until allocations and the strict rolling
MFU gate are observed from live job evidence.
