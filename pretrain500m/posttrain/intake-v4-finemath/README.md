# Bounded FineMath intake v4

This tranche reads four deterministic, previously unused FineMath row groups
to cover the measured F2 two-epoch deficit with filtering and family-collision
reserve. It remains outside training until the existing quality,
decontamination, family-disjointness, tokenization, and manifest gates pass.

The sampler binds the completed intake-v2 receipt and excludes the transitive
union of all v1 and v2 physical groups. Network reads remain capped at 128 MiB,
uncompressed output at 192 MiB, and output rows at 100,000. The disk floor is
the frozen aggregate gate for four active model-only writers plus 256 MiB, so
this CPU-side acquisition cannot consume their reserved checkpoint headroom.
