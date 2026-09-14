#!/usr/bin/env python3
from __future__ import annotations

import unittest

from build_time_matched_binding_manifest import duplicate_train_rows


class TimeMatchedManifestTests(unittest.TestCase):
    def test_only_train_is_duplicated_and_ids_are_unique(self) -> None:
        rows = [
            {
                "split": "train",
                "scene_family_id": "family",
                "scene_pair_id": "pair",
                "statistical_cluster_id": "pair",
            },
            {
                "split": "mechanism_dev",
                "scene_family_id": "dev-family",
                "scene_pair_id": "dev-pair",
                "statistical_cluster_id": "dev-pair",
            },
        ]
        output = duplicate_train_rows(rows, 2)
        train = [row for row in output if row["split"] == "train"]
        self.assertEqual(len(train), 2)
        self.assertEqual(len({row["scene_family_id"] for row in train}), 2)
        self.assertTrue(all(row["scene_pair_id"] == row["statistical_cluster_id"] for row in train))
        self.assertEqual(sum(row["split"] == "mechanism_dev" for row in output), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
