# DCLM top-up confirmation launcher

This launcher waits for the hash-bound top-up pipeline receipt, reruns target
training and builder tests, builds the unchanged F2/F3 long protocol against the
incremental artifacts, and invokes the existing GPU-only submission gate.
