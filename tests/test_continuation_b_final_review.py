from finalize_continuation_b_review import MORE,reject_reason
from refine_continuation_b_review import EXCLUDED


def test_expanded_review_preserves_previous_exclusions():
    for identifier,reason in {**MORE,**EXCLUDED}.items():
        assert reject_reason('dclm',identifier,'placeholder',1000)==reason
