from hashlib import sha256
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from audit_pixmo_points_eval_image_availability_v1 import deterministic_sample, verify_image_bytes


class PixMoPointsEvalImageAvailabilityTests(unittest.TestCase):
    def test_sample_is_unique_hash_sorted_and_model_blind(self):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "fixture.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"image_url": "https://example/c", "image_sha256": "c" * 64},
            {"image_url": "https://example/a", "image_sha256": "a" * 64},
            {"image_url": "https://example/a", "image_sha256": "a" * 64},
            {"image_url": "https://example/b", "image_sha256": "b" * 64},
        ]), path)
        try:
            sample = deterministic_sample(path, 2)
            self.assertEqual([row["image_sha256"] for row in sample], ["a" * 64, "b" * 64])
        finally:
            directory.cleanup()

    def test_hash_and_decode_must_both_pass(self):
        buffer = BytesIO()
        Image.new("RGB", (3, 2), "red").save(buffer, format="PNG")
        payload = buffer.getvalue()
        result = verify_image_bytes(payload, sha256(payload).hexdigest(), 10000)
        self.assertEqual(result["status"], "verified")
        self.assertEqual((result["width"], result["height"]), (3, 2))
        mismatch = verify_image_bytes(payload, "0" * 64, 10000)
        self.assertEqual(mismatch["status"], "sha256_mismatch")

    def test_invalid_image_with_matching_hash_fails_decode(self):
        payload = b"not-an-image"
        result = verify_image_bytes(payload, sha256(payload).hexdigest(), 10000)
        self.assertEqual(result["status"], "decode_failure")

    def test_size_limit_is_fail_closed(self):
        payload = b"x" * 11
        result = verify_image_bytes(payload, sha256(payload).hexdigest(), 10)
        self.assertEqual(result["status"], "size_limit_exceeded")

    def test_hash_mapping_conflict_fails(self):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "fixture.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"image_url": "https://example/a", "image_sha256": "a" * 64},
            {"image_url": "https://example/b", "image_sha256": "a" * 64},
        ]), path)
        try:
            with self.assertRaisesRegex(RuntimeError, "multiple URLs"):
                deterministic_sample(path, 1)
        finally:
            directory.cleanup()


if __name__ == "__main__":
    unittest.main()
