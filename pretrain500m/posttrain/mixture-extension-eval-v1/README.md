# R4 extension two-seed capability screen v1

After both 1,073,741,824-token R4 extensions complete, `build_plan.py` verifies
their Slurm exits, status, checkpoint hashes, source identities, and exact seeds
before freezing the two-candidate evaluation plan. The plan binds the already
matched two-RTX-5090 Base result and the same seven English base-model tasks.

Each independent evaluation job uses two physical RTX 5090 GPUs, verifies the
offline cache and native checkpoint, requires bitwise export state parity and
bounded BF16 logit drift, and records per-example evaluation logs. Its temporary
Hugging Face export lives under node-local `/tmp` and is deleted after the
result receipt is written. `submit.py` checks for duplicates and available disk
space before submitting both seeds exactly once.

The seven-task protocol was used during candidate selection, so this is an
adaptive retention and capability screen. It cannot establish sealed
performance, formal promotion, instruction following, multilingual generation,
safety, or market superiority.
