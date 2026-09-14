import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from acquire_openimages_posture_confirmation_captions_v1 import (
    validate_jsonl,
    validate_protocol,
)


class ConfirmationCaptionAcquisitionTests(unittest.TestCase):
    def test_jsonl_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.jsonl"
            path.write_text(
                json.dumps({"image_id": "a", "caption": "A boy is sitting."}) + "\n"
                + json.dumps({"image_id": "b", "caption": "A woman is standing."}) + "\n"
            )
            source = {
                "expected_bytes": path.stat().st_size,
                "expected_md5_hex": hashlib.md5(path.read_bytes()).hexdigest(),
                "expected_rows": 2,
                "expected_unique_image_ids": 2,
            }
            record = validate_jsonl(path, source)
            self.assertEqual(record["rows"], 2)
            self.assertEqual(record["unique_image_ids"], 2)

    def test_protocol_blocks_evaluation(self):
        protocol = {
            "status": "authorized_openimages_validation_caption_acquisition_only_v1",
            "download_authorized": True,
            "training_authorized": False,
            "model_evaluation_authorized": False,
            "automatic_training_start": False,
            "automatic_evaluation_start": False,
        }
        validate_protocol(protocol)
        protocol["model_evaluation_authorized"] = True
        with self.assertRaisesRegex(RuntimeError, "model_evaluation"):
            validate_protocol(protocol)


if __name__ == "__main__":
    unittest.main()
