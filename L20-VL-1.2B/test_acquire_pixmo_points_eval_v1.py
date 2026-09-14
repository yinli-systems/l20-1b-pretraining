import tempfile
from pathlib import Path
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from acquire_pixmo_points_eval_v1 import profile_parquet, verify_source_file


class PixMoPointsEvalAcquisitionTests(unittest.TestCase):
    def write_fixture(self, rows):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "fixture.parquet"
        schema = pa.schema([
            ("image_url", pa.string()),
            ("image_sha256", pa.string()),
            ("label", pa.string()),
            ("points", pa.list_(pa.struct([("x", pa.float64()), ("y", pa.float64())]))),
            ("masks", pa.list_(pa.list_(pa.list_(pa.bool_())))),
        ])
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
        return directory, path

    def test_profile_records_integrity_counts(self):
        sha = "a" * 64
        directory, path = self.write_fixture([
            {"image_url": "https://example/a", "image_sha256": sha, "label": "item", "points": [{"x": 10.0, "y": 20.0}], "masks": [[[True]]]},
            {"image_url": "https://example/a", "image_sha256": sha, "label": "none", "points": [], "masks": [[[False]]]},
        ])
        try:
            result = profile_parquet(path)
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["unique_image_sha256"], 1)
            self.assertEqual(result["total_points"], 1)
            self.assertEqual(result["rows_with_zero_points"], 1)
            self.assertEqual(result["rows_where_point_and_mask_counts_differ"], 1)
        finally:
            directory.cleanup()

    def test_out_of_range_coordinate_fails(self):
        directory, path = self.write_fixture([
            {"image_url": "https://example/a", "image_sha256": "a" * 64, "label": "item", "points": [{"x": 101.0, "y": 20.0}], "masks": [[[True]]]}
        ])
        try:
            with self.assertRaisesRegex(RuntimeError, "outside"):
                profile_parquet(path)
        finally:
            directory.cleanup()

    def test_invalid_hash_fails(self):
        directory, path = self.write_fixture([
            {"image_url": "https://example/a", "image_sha256": "bad", "label": "item", "points": [{"x": 1.0, "y": 2.0}], "masks": [[[True]]]}
        ])
        try:
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                profile_parquet(path)
        finally:
            directory.cleanup()

    def test_staged_source_integrity(self):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "source.bin"
        path.write_bytes(b"abc")
        try:
            verify_source_file(path, {
                "bytes": 3,
                "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            })
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                verify_source_file(path, {"bytes": 3, "sha256": "0" * 64})
        finally:
            directory.cleanup()


if __name__ == "__main__":
    unittest.main()
