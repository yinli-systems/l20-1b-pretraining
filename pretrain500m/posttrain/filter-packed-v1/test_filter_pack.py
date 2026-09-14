import numpy as np
import pytest

from filter_pack import span


def test_span_reads_array_tail_boundary():
    array = np.arange(8, dtype=np.uint16)
    tail = [8, 9, 10]
    assert np.concatenate(span(array, tail, 2, 5)).tolist() == [2, 3, 4]
    assert np.concatenate(span(array, tail, 6, 10)).tolist() == [6, 7, 8, 9]
    assert np.concatenate(span(array, tail, 8, 11)).tolist() == [8, 9, 10]


def test_span_rejects_invalid_offsets():
    with pytest.raises(ValueError):
        span(np.arange(3, dtype=np.uint16), [3], 2, 5)
