# Declared quality filters and family features

This stage applies explicit screening filters to actual raw documents, combines
the bound conservative exclusions, and writes family and MinHash features. It
does not create a training admission receipt. Four CPU workers reuse the formal
tokenizer's hash-bound row index; no document is tokenized again or head-truncated.
The original raw data and text remain intact.

The policy in `features.py` uses minimum document sizes, control/replacement
characters, repeated long lines, and dominant-character fractions. It retains
code/math symbols and code indentation. Code must have the already required
permissive license and education score; Python is parsed, never executed. Math
retains its upstream 4+ threshold. Natural-language data is checked against
source labels/scores and start/middle/end language windows. Numerical thresholds
are project screening choices, not research-proven universal quality cutoffs.
Language probability is not a factual or educational-quality score. Factual
correctness, manual content review and domain usefulness remain unverified.

Language identification uses the full 176-language fastText model, downloaded
from its [official distribution](https://fasttext.cc/docs/en/language-identification.html),
frozen by SHA256. The model is distributed under CC-BY-SA 3.0; credit fastText's
authors (Joulin, Grave, Bojanowski and Mikolov). Its Windows/macOS compatibility is
not claimed: the pinned wheel is for the real CPython 3.12 Linux runtime. The
adapter uses the wrapper's same native prediction call to avoid its NumPy 2
`copy=False` conversion issue without altering installed packages.

URL features preserve path case and encoded separators, canonicalize ordinary
www/http/tracking aliases, and use the frozen
[Public Suffix List](https://publicsuffix.org/list/) including private rules,
wildcards and exceptions. The list retains its MPL-2.0 notice. Host IDNA conversion
uses Python's built-in codec; canonical keys do not prove all URL aliases resolved.
Code repositories and exact numeric templates receive separate identity keys.

Full-text Unicode-aware token 5-shingles feed 64 deterministic MinHash transforms
in bounded NumPy chunks. The arithmetic uses universal hashing with unsigned
integer wrapping, as in the [datasketch reference implementation](https://github.com/ekzhu/datasketch/blob/master/datasketch/minhash.py).
The resulting 16 bands of four values are candidate indexes, not proof of a
similarity threshold. Hash collisions, approximate recall, paraphrases, translations
and identifier renaming limit detection. Actual candidate comparison, duplicate
representative choice, family closure and train/dev/confirmation assignment remain
separate steps; repository forks are not resolved by lowercasing names alone.

Exclusions combine the three separately verified supplemental benchmark spans,
181 legacy benchmark hash candidates, and matches against the complete old source
snapshot. The latter two remain conservative policy exclusions, not upgraded to
verified benchmark spans or exact final-pack membership. Feature eligibility
consumes this union; no training pack has yet consumed it.

Twelve fixture checks cover URL/PSL boundaries, code licenses/syntax, Unicode,
chunk-invariant and scalar-reference signatures, intact math/code text, wrong-language/repeated/garbled
text, raw-index identity, exclusion application and real native language inference.
The native inference check requires the target runtime and is explicitly skipped
locally when the target-only wheel is unavailable.
