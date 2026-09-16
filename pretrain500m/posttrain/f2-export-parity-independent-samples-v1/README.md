# Independent-sample strict export diagnostic

This diagnostic freezes two new random input seeds before execution and applies
the original BF16 HF reload gate without changing its thresholds. Both F2
continuation checkpoints run on both seeds under the same eager-attention,
deterministic-algorithm, 2x128-token setup.

Every candidate writes its bitwise tensor-parity and logit-drift diagnostic
before the original gate can raise. The wrapper retains all four diagnostics
and exits nonzero if any candidate fails the unchanged gate. This experiment
tests whether the earlier 247/256 argmax result is sample-sensitive. It does
not change the failed result, establish a new export protocol, or measure model
capability.
