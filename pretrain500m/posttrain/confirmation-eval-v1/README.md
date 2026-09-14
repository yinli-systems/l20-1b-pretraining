# Held-out confirmation evaluation v1

This evaluates the frozen base checkpoint and both selected F3 screen seeds on
the family-disjoint confirmation pack. It uses exact document loss masks and an
equal mean across general web, knowledge, math, code, and multilingual domains.
The gate requires both F3 seeds to improve overall confirmation loss versus the
base and permits at most 0.5% relative loss regression in any domain.

This is confirmation of the short screen direction. It does not replace the
planned longer confirmation training or generative task evaluation.
