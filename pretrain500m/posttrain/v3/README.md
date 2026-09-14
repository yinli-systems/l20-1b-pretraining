# Mixture runner with the validated recovery policy

This candidate carries the stable DDP reduction policy from recovery-v2 into
the full corpus-admission runner. Job 1588249 completed the actual four-RTX-5090,
529M-model test with `PASS_GPU_RECOVERY_EXACT`: 32 continuous steps matched
16 steps plus a new process and 16 resumed steps across model, optimizer,
reader, origin, run fingerprint and all-rank RNG state. Both steady MFU windows
were about 77.5%. The source comparison remains an engineering result on the
original FineWeb corpus, not a new-mixture quality result.

`find_unused_parameters=True`, `static_graph=False`, the 25 MiB bucket cap and
gradient bucket views match the qualified policy. The policy participates in
the immutable run fingerprint; unexpected bucket rebuilding stops the runner.
The full corpus-admission block is unchanged byte-for-byte from v2 and still
requires license, quality, cross-source deduplication, benchmark decontamination
and family-disjoint split checks, all bound to the actual data/protocol hashes.
No engineering-only receipt can admit a new corpus through this runner.

The original mixture and resume modules and CPU regression suite are unchanged.
Recovery-v2's heavy diagnostic boundary hashes are omitted from this production
candidate, but exact resume identity and the MFU gate remain. Runtime and
performance with actual admitted multi-source packs remain to be qualified.
Formal mixture training has not yet launched.
