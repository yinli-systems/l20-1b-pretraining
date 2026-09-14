# Supplemental contamination scan for confirmation expansion

This source preserves the frozen Unicode exact-span policy and benchmark bundle while replacing the intake preflight with the generalized, physical-row-disjoint expansion audit. Candidate hashes remain conservative exclusions; the scan does not admit training data.

`--scan-from-tranche` still binds and verifies every historical input but scans only appended tranches. Use it only when the exclusion builder starts from the already frozen exclusion union for the skipped prefix.
