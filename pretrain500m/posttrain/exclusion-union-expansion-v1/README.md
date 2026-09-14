# Incremental exclusion union

This builder starts from a frozen exclusion plan for the historical prefix and
adds every exact-span, legacy 13-word hash, and old-corpus normalized-overlap
candidate found in appended tranches. It binds the reports and every candidate
artifact by SHA256. It does not modify raw text or grant training admission.
