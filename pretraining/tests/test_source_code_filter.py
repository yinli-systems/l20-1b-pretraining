from source_docs import _code_row_is_eligible


def test_code_filter_uses_release_validated_language_thresholds():
    assert _code_row_is_eligible("Python", {"license_type": "permissive", "int_score": 3})
    assert not _code_row_is_eligible("Python", {"license_type": "permissive", "int_score": 2})
    assert _code_row_is_eligible("Java", {"license_type": "permissive", "int_score": 2})


def test_code_filter_remains_fail_closed_on_license():
    assert not _code_row_is_eligible("Python", {"license_type": "no_license", "int_score": 5})
    assert not _code_row_is_eligible("Java", {"int_score": 5})
