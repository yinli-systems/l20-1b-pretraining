import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = Path(__file__).with_name("build_inputs.py")
SPEC = importlib.util.spec_from_file_location("new_pool_build_inputs", MODULE_PATH)
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class NewPoolBuildTests(unittest.TestCase):
    def test_all_recipes_close_exactly_and_preserve_source_set(self):
        expected = set(BUILD.RECIPES_BPS["N0_new_pool_broad"])
        for recipe, basis_points in BUILD.RECIPES_BPS.items():
            self.assertEqual(set(basis_points), expected, recipe)
            self.assertEqual(sum(basis_points.values()), 10_000, recipe)
            quotas = BUILD.exact_block_quotas(basis_points)
            self.assertEqual(sum(quotas.values()), BUILD.TARGET_BLOCKS, recipe)
            for source, bps in basis_points.items():
                exact = BUILD.TARGET_BLOCKS * bps / 10_000
                self.assertLessEqual(abs(quotas[source] - exact), 1, (recipe, source))

    def test_seven_arms_cover_five_recipes_and_balanced_lr_grid(self):
        self.assertEqual(len(BUILD.ARMS), 7)
        self.assertEqual({recipe for _, recipe, _ in BUILD.ARMS}, set(BUILD.RECIPES_BPS))
        grid = {lr for _, recipe, lr in BUILD.ARMS if recipe == "N3_new_pool_balanced"}
        self.assertEqual(grid, {"0.00003", "0.00006", "0.0001"})

    def test_packed_array_shape_and_dtype_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.npy"
            metadata = {
                "blocks": 2,
                "array_tokens": 2 * (BUILD.SEQUENCE_LENGTH + 1),
                "prediction_tokens": 2 * BUILD.SEQUENCE_LENGTH,
                "dtype": "uint16",
            }
            np.save(path, np.zeros(metadata["array_tokens"], dtype=np.uint16))
            BUILD.validate_packed_array(path, metadata, "fixture")
            np.save(path, np.zeros(metadata["array_tokens"] - 1, dtype=np.uint16))
            with self.assertRaisesRegex(ValueError, "packed array mismatch"):
                BUILD.validate_packed_array(path, metadata, "fixture")


if __name__ == "__main__":
    unittest.main()
