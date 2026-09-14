import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from audit_cosyn_point_source_acquisition_v1 import REQUIRED_COLUMNS, validate_source
from train_stage_a_full_token import sha256_file


class CoSynPointSourceAcquisitionAuditTests(unittest.TestCase):
    def test_required_schema_is_exact(self):
        self.assertEqual(REQUIRED_COLUMNS, {"id", "image", "questions", "answer_points", "names"})

    def test_validate_source_rejects_size_mismatch_before_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.parquet"
            table = pa.table({
                "id": ["a"],
                "image": [{"bytes": b"x", "path": None}],
                "questions": [["q"]],
                "answer_points": [[{"x": [50.0], "y": [50.0]}]],
                "names": [["n"]],
            })
            pq.write_table(table, path)
            with self.assertRaises(RuntimeError):
                validate_source(path, {"expected_bytes": path.stat().st_size + 1})

    def test_validate_source_reports_hash_and_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.parquet"
            table = pa.table({
                "id": ["a", "b"],
                "image": [{"bytes": b"x", "path": None}, {"bytes": b"y", "path": None}],
                "questions": [["q"], ["r"]],
                "answer_points": [[{"x": [50.0], "y": [50.0]}], [{"x": [25.0], "y": [75.0]}]],
                "names": [["n"], ["m"]],
            })
            pq.write_table(table, path)
            result = validate_source(path, {"expected_bytes": path.stat().st_size})
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["unique_ids"], 2)
            self.assertEqual(result["sha256"], sha256_file(path))


if __name__ == "__main__":
    unittest.main()
