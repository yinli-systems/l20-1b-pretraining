from pathlib import Path
import tempfile
import unittest

from PIL import ImageFont

from build_highres_evidence_search_pilot_v1 import (
    build_family,
    center_patch_index,
)


class HighresEvidenceSearchBuilderTests(unittest.TestCase):
    FONT_CANDIDATES = (
        Path("/System/Library/Fonts/SFNS.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    )

    @classmethod
    def font(cls):
        for path in cls.FONT_CANDIDATES:
            if path.exists():
                return ImageFont.truetype(str(path), 12)
        raise RuntimeError("no test font is available")

    def test_patch_index_uses_box_center(self):
        self.assertEqual(center_patch_index((0, 0, 100, 100), 1000, 10), 0)
        self.assertEqual(center_patch_index((900, 900, 1000, 1000), 1000, 10), 99)
        self.assertEqual(center_patch_index((400, 500, 500, 600), 1000, 10), 54)

    def test_local_counterfactual_changes_answer_but_not_address(self):
        directory = tempfile.TemporaryDirectory()
        try:
            font = self.font()
            rows = build_family(0, "development", "read_local_digit", 7, Path(directory.name), 256, 14, font, font)
            self.assertEqual(len(rows), 2)
            self.assertNotEqual(rows[0]["answer"], rows[1]["answer"])
            self.assertEqual(rows[0]["target_patch_index_14x14"], rows[1]["target_patch_index_14x14"])
            self.assertEqual(rows[0]["question"], rows[1]["question"])
            self.assertEqual(rows[0]["expected_action"], "POINT")
        finally:
            directory.cleanup()

    def test_global_counterfactual_changes_answer_and_stays_stop(self):
        directory = tempfile.TemporaryDirectory()
        try:
            font = self.font()
            rows = build_family(1, "development", "read_global_border", 7, Path(directory.name), 256, 14, font, font)
            self.assertNotEqual(rows[0]["answer"], rows[1]["answer"])
            self.assertEqual([row["expected_action"] for row in rows], ["STOP", "STOP"])
            self.assertEqual([row["target_crop_box_xyxy_normalized"] for row in rows], [None, None])
        finally:
            directory.cleanup()

    def test_invalid_box_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "invalid target box"):
            center_patch_index((-1, 0, 10, 10), 100, 10)


if __name__ == "__main__":
    unittest.main()
