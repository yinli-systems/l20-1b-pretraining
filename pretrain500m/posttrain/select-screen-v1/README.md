# Frozen two-seed CPT screen selector

This tool consumes the exact eight F0/F1/F2/F3 screen directories. It verifies
base, protocol, validation, budget, DDP policy, two complete seeds, finite
metrics, the MFU gate, validation coverage, checkpoint status, and the absence
of bucket rebuilds. It ranks F1/F2/F3 by lower worst-seed equal-domain loss,
then lower mean loss, and requires the winner to beat the paired F0 control for
both seeds.

The resulting winner remains a screen result. Held-out confirmation and an
expanded admitted training corpus are separate gates before a longer run.
