# Corpus preparation prerequisites

These local tools extend the research into exact subsource budgets and inspect
the available family metadata. They do not emit a passing corpus-admission
receipt or launch training.

## Exact subsource budgets

`plan_subsources.py` first allocates the frozen top-level recipe, then apportions
each parent's exact integer block budget among its children. This preserves
the research planner's top-level totals; flattening all percentages before
rounding could change them. Both the design and dataset inventory are hashed.
The output covers all four screen options and all possible confirmation options.
Only two selected candidate recipes and the control would enter confirmation.

The frozen internal weights are FineMath/InfiWebMath 75/25; Python/JavaScript/
TypeScript/Cpp/Java 50/20/10/10/10; and Chinese/Spanish/French/German/Japanese/
Arabic 50/10/10/10/10/10. These are experimental choices, not measured optima.
Leaf quotas are expressed in complete packed blocks and prediction tokens, with
realized percentages recorded after rounding. They still need actual admitted
content and measured leaf provenance in the materialized parent pools or a
hierarchical sampler. The v2 reader alone only controls top-level groups.

```sh
python3 pretrain500m/posttrain/admission/plan_subsources.py --output pretrain500m/posttrain/admission/subsource-quotas-v1.json
```

## Current sample-family audit

`audit_sample_families.py` examines 1,038 existing pretraining inspection rows.
It groups observed canonical URL keys, code repository names and synthetic
prompts, while separately checking normalized exact text identity. It does not
make a train/dev split or consume benchmark contents. The samples were selected
for inspection within a small number of row groups; their host distribution
must not be extrapolated to the full corpora.

| Development domain | Observed family-key upper bound | Required independent families |
|---|---:|---:|
| General web | 256 | 1,000 |
| Knowledge/reading | 128 | 1,000 |
| Math | 253 | 1,000 |
| Code | 14 | 1,000 |
| Multilingual | 256 | 1,000 |

The counts can decrease when mirrors, near duplicates and shared parents are
joined. All five domains are insufficient even before those audits. No normalized
exact duplicate text was found across this small pool; that is not a global
deduplication result. All 128 Cosmopedia rows lack an explicit seed-document
identity in the retained samples. Prompt hashes alone cannot certify parent
independence. The original sampling column list also omitted `seed_data`; new
acquisition must inspect the full source metadata and determine whether an actual
parent identifier can be recovered, rather than assuming the field name proves it.

The existing `pretrain500m/build_decontam.py` uses `[a-z0-9]+` and English-style
13-word n-grams. That does not supply the multilingual/code contamination audit
required by the new program. Its historical receipt must not be relabeled as
passing coverage for the new domains. A new audit must specify its Unicode/code
normalization, preserved fields, benchmark scope, and overlap limitations.

```sh
python3 pretrain500m/posttrain/admission/audit_sample_families.py --output pretrain500m/posttrain/admission/sample-family-audit-v1.json
```

Next acquisition should retain source revision and physical row location,
canonical URL/repository identity, synthetic parent metadata, language/quality
scores, code license evidence, and text-content hashes. Family clustering and
split assignment must precede packing. Corpus admission, actual subsource
materialization and GPU qualification remain pending.

## Combined raw-tranche audit

`audit_raw_intake_v2.py` scans completed tranches together in the formal tokenizer.
Pass `--input` once per tranche, oldest first. Every tranche must have its final
summary, zero blocked sources, matching raw/receipt file sets and no writer lock.
It verifies input hashes, source revisions, the receipt chain, excluded groups,
physical group disjointness and every retained row's physical location. Bound
metadata, raw files, tokenizer and quotas are rechecked after the whole scan.

Exact content ownership is deterministic by source ID, then tranche argument
order, then row. The report separates per-source unique upper bounds from global
deduplicated assigned quotas; cross-tranche/source duplicates are not counted
twice. Normalized nonidentical candidates remain available for review. The row
index keeps original provenance and exact-duplicate ownership. This measures raw
content only: independent families, synthetic parents, near duplicates, old
FineWeb overlap, contamination, quality filters and held-out reserves still need
work. The output never constitutes corpus admission.

The owned-fixture regression suite covers cross-tranche and cross-source duplicate
accounting, case-normalized nonidentical text retention, provenance/row failures,
modified inputs, incomplete writers and changed prior receipt identities. It
does not substitute for running this audit on the real combined data.
