import hashlib
import tempfile
from pathlib import Path
import unittest

from acquire_openimages_posture_confirmation_v1 import validate_file, validate_protocol


class OpenImagesConfirmationAcquisitionTests(unittest.TestCase):
    def valid_protocol(self):
        return {
            "status": "authorized_openimages_validation_annotation_acquisition_only_v1",
            "download_authorized": True,
            "training_authorized": False,
            "model_evaluation_authorized": False,
            "automatic_training_start": False,
            "automatic_evaluation_start": False,
            "sources": {
                "validation_visual_relationships": {},
                "validation_image_metadata": {},
            },
        }

    def test_protocol_is_fail_closed(self):
        validate_protocol(self.valid_protocol())
        changed = self.valid_protocol()
        changed["model_evaluation_authorized"] = True
        with self.assertRaisesRegex(RuntimeError, "model_evaluation"):
            validate_protocol(changed)

    def test_file_size_and_md5(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_bytes(b"a,b\n1,2\n")
            expected = {
                "expected_bytes": path.stat().st_size,
                "expected_md5_hex": hashlib.md5(path.read_bytes()).hexdigest(),
            }
            record = validate_file(path, expected)
            self.assertEqual(record["bytes"], 8)
            self.assertEqual(len(record["sha256"]), 64)

    def test_file_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_bytes(b"content")
            with self.assertRaisesRegex(RuntimeError, "MD5"):
                validate_file(
                    path,
                    {"expected_bytes": 7, "expected_md5_hex": "0" * 32},
                )


if __name__ == "__main__":
    unittest.main()
