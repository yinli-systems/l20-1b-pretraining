# Targeted extension after the combined raw audit

The completed first two tranches contain 180,407 rows and 180,374 byte-identical
unique documents. Their globally assigned packed prediction-token upper bound is
276,158,464 before quality filters, benchmark exclusions and held-out reserves.
Only PDF, DCLM and the unresolved synthetic source have positive raw lower-bound
shortages for the largest single screen quota at two epochs: 8,241,152;
5,993,472; and 10,462,208 tokens respectively. These are not cumulative
screening/confirmation sufficiency claims.

This extension fetches only PDF and DCLM. The plan uses the measured shortage
times 1.35 plus an explicit 8,388,608-token planning reserve per source, divided
by observed retained tokens per consumed physical group. The reserve is an
assumption for acquisition sizing, not a measured retention or family guarantee.
Resulting group requests and prior receipt identities are in `segments.json`.
Cosmopedia is not expanded while its source parents remain unresolved.

The sampler verifies the completed intake-v2 receipt hash and source identity.
`exclude_row_groups` retains immediate-prior groups, preserving the existing
combined auditor's linked-receipt convention. `all_excluded_row_groups` separately
binds the full transitive union of every consumed earlier group, including capped
or partially retained groups. Sampling excludes that full union. Thus the frozen
auditors can verify all three tranches without modifying their source: immediate
receipt links and global physical disjointness are both checked.

The seven owned-fixture checks cover immediate and transitive exclusions,
changed receipts/source identity, and the actual frozen auditor accepting three
disjoint tranches. The old two outputs remain immutable. Limits remain two source
workers, 128 MiB Parquet reads and 192 MiB uncompressed output per source, 100,000
rows per source and an 18 GiB free-space floor. Outputs are raw and unadmitted.
Once complete, remeasure all three tranches together and apply both supplemental
exclusion hashes plus remaining legacy, family, quality and old-corpus checks
before packing or training.
