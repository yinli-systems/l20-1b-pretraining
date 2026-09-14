from audit import candidate_reasons, repeated_ngram_coverage


def test_repeated_ngram_coverage_detects_repeated_sequence():
    tokens = list("abcdefghij") * 5
    coverage, maximum = repeated_ngram_coverage(tokens)
    assert coverage == 1.0
    assert maximum >= 4


def test_repeated_ngram_coverage_does_not_flag_unique_sequence():
    coverage, maximum = repeated_ngram_coverage([str(i) for i in range(30)])
    assert coverage == 0.0
    assert maximum == 1


def test_known_injection_candidates_are_review_only():
    spanish = "ensayo académico " * 100 + " porno puta prostituta escort follar webcam porno puta prostituta escort"
    reasons = candidate_reasons("multilingual_spa_Latn", spanish, 0.0, 0.9)
    assert "adult_keyword_stuffing" in reasons
    japanese = "本文" + "盛岡の出会い掲示板" * 3
    assert "repeated_dating_keyword" in candidate_reasons("multilingual_jpn_Jpan", japanese, 0.0, 0.9)
