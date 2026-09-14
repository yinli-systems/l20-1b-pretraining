import hashlib
import tempfile
import unittest
from pathlib import Path

from acquire_openimages_vrd import validate_file


class OpenImagesVRDAcquisitionTests(unittest.TestCase):
    def test_validate_file_records_all_hashes(self):
        payload = b"bounded-source\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_bytes(payload)
            source = {
                "expected_bytes": len(payload),
                "expected_md5_hex": hashlib.md5(payload).hexdigest(),
            }
            record = validate_file(path, source)
            self.assertEqual(record["bytes"], len(payload))
            self.assertEqual(record["md5"], hashlib.md5(payload).hexdigest())
            self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())

    def test_validate_file_fails_closed_on_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_bytes(b"abc")
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                validate_file(path, {"expected_bytes": 4, "expected_md5_hex": "ignored"})


if __name__ == "__main__":
    unittest.main()
