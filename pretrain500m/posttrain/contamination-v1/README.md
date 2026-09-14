# Supplemental Unicode exact-span contamination audit

This scanner finds candidate exclusions against the separately reserved benchmark
bundle. It does not delete corpus rows or issue corpus admission. The old seven
tasks, original FineWeb train/validation overlap, near duplicates, families and
quality checks remain separate requirements. Scans cover only named completed
intake tranches; a first-tranche scan is not a combined-corpus result.

Normalization uses NFC and collapses Unicode whitespace. Prose casefolds; code
and mathematics retain case. Digits, identifiers, punctuation and operators are
retained. Compatibility folding is deliberately absent. Whitespace collapsing
also changes Python indentation: a match is evidence of shared content, not
equivalent program behavior. No benchmark code is executed.

Reference fields are handled separately. A complete normalized field of 64–192
characters becomes one anchor. Longer fields contribute 192-character anchors at
stride 96 plus a tail anchor. Anchors need at least 12 distinct nonspace
characters. A contiguous long-field overlap of at least 287 normalized characters
contains an anchor before the diversity filter; smaller partial overlaps can be
missed. This is a declared exact-span scope, not a paraphrase, translation,
identifier-renaming or semantic contamination detector.

Short primary prompts/questions/problems can match a whole document only, with
at least 16 characters and eight distinct nonspace characters. Common short
answers, choices and low-diversity fields are skipped and counted. Each reference
file receives a coverage report, so omitted fields cannot be hidden by a single
aggregate pass label. Saved results include source row provenance, content hash,
normalized span offsets and benchmark owner identities. At most 32 distinct
pattern hits and four owners per hit are saved; truncation/owner counts are
explicit and the pinned reference bundle permits reconstruction.

Matching uses the Unicode build of
[pyahocorasick 2.3.1](https://pypi.org/project/pyahocorasick/2.3.1/).
Its [documented automaton iterator](https://pyahocorasick.readthedocs.io/en/latest/)
returns exact matching positions; the scanner independently compares every saved
substring to its reference anchor. The CPython 3.12 Linux wheel is pinned by
SHA256 in `dependency.json` and installed into a dedicated project overlay.
Existing runtime packages are not replaced.

The regression suite uses owned text to exercise Chinese/Japanese/Arabic,
canonical Unicode composition, whitespace and prose case changes, code case,
operators and numeric changes, superscript distinctions, short-answer controls,
field boundaries, long-span coverage gaps, hit limits and reference corruption.
Real reference indexing and corpus scans are recorded separately from these
fixture checks. Candidate exclusions still require interpretation before packing.
