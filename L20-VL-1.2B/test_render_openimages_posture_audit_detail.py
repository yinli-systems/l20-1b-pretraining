import unittest

from PIL import Image

import render_openimages_posture_audit_detail as module


class PostureDetailAuditRendererTests(unittest.TestCase):
    def test_crop_with_context_is_nonempty_and_bounded(self):
        image = Image.new("RGB", (100, 80), "white")
        crop = module.crop_with_context(image, [0.0, 0.2, 0.8, 1.0])
        self.assertGreater(crop.width, 0)
        self.assertGreater(crop.height, 0)
        self.assertLessEqual(crop.width, image.width)
        self.assertLessEqual(crop.height, image.height)

    def test_draw_box_preserves_size(self):
        image = Image.new("RGB", (100, 80), "white")
        output = module.draw_box(image, [0.1, 0.9, 0.2, 0.8])
        self.assertEqual(output.size, image.size)


if __name__ == "__main__":
    unittest.main()
