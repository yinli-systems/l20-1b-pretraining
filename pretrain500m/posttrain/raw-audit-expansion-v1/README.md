# Combined raw audit for confirmation expansion

This version combines the original admitted raw tranches, the completed DCLM/PDF/FineMath top-ups, and generalized expansion segments. It binds receipt and raw-file hashes, validates exact physical row-group disjointness including generalized history unions, assigns a unique file ordinal when several segments share one logical source, tokenizes with the frozen tokenizer, and reports per-source unique-token sufficiency against the F2/F3 2.147B-token confirmation plan. It does not grant training admission.

`audit_incremental.py` additionally binds an already completed prefix audit and its row index, re-hashes every historical input, re-keys its rows to the unique file ordinal, and tokenizes only appended tranches. Historical exact-dedup ownership is append-stable. This avoids repeating the expensive tokenizer pass while retaining a combined index for downstream quality and family scans.

Chained incremental audits map every historical generalized file ordinal through
its bound absolute path, so one prior input root may contain multiple physical
files for the same logical source without aliasing tranche identities.
