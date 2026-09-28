import threading
from concurrent.futures import ThreadPoolExecutor

from source_docs import _rolling_unordered_map


def test_rolling_map_does_not_head_of_line_block_on_first_item():
    release_first = threading.Event()
    started: list[int] = []
    started_lock = threading.Lock()

    def work(value: int) -> int:
        with started_lock:
            started.append(value)
        if value == 0:
            assert release_first.wait(timeout=5)
        return value

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = _rolling_unordered_map(executor, work, iter(range(20)), max_in_flight=8)
        first = next(results)
        assert first != 0
        release_first.set()
        remaining = list(results)

    assert sorted([first, *remaining]) == list(range(20))


def test_rolling_map_rejects_nonpositive_window():
    with ThreadPoolExecutor(max_workers=1) as executor:
        results = _rolling_unordered_map(executor, lambda value: value, iter(()), max_in_flight=0)
        try:
            next(results)
        except ValueError as error:
            assert str(error) == "max_in_flight must be positive"
        else:
            raise AssertionError("expected ValueError")
