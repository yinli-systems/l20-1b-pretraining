from refine_continuation_b_review import EXCLUDED,reject_reason


def test_observed_bad_examples_fail_closed():
    for identifier,reason in EXCLUDED.items():
        assert reject_reason('dclm',identifier,'some text',1000)==reason
    assert reject_reason('narrative','book','Notice: [missing text]',3000)=='declared_missing_source_text'
    assert reject_reason('narrative','book','abc\u0093'*11,3000)=='repeated_c1_encoding_artifacts'
