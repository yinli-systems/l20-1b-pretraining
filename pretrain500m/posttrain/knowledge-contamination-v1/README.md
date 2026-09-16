# SciQ current-corpus contamination gate

This frozen scan runs before any SciQ model scoring. It checks all 199,407 raw
rows in the three intake tranches, a conservative superset of F2's selected
training documents. It detects normalized exact questions plus fixed lexical
5-token question/option and 13-token support shingle coverage. Any affected
SciQ row is removed from both proxy roles.

This policy does not detect semantic paraphrases or model-ancestor exposure.
Passing it establishes only a current-corpus exact and lexical-near gate. At
least 800 of 1,000 rows must remain, after which the nine known duplicate-choice
rows are also excluded before the output-blind role split.
