# DDP speed benchmark v1

This is a same-allocation ABBA comparison of DDP
`find_unused_parameters=true` and `false`. Every segment uses the same four
RTX 5090 GPUs, seed, model initialization, F3 data, optimizer, learning-rate
schedule, and 20-step token budget. Steps 1 through 5 are excluded from the
steady-state median. The runner writes metrics and validation only.

The faster path is admitted only when every segment keeps its final ten-step
median MFU strictly above 0.50, the loss and gradient traces match within
1e-12, and the paired median speedup exceeds 0.5%. A pass changes only a future
runner; it never changes or restarts active screen jobs.
