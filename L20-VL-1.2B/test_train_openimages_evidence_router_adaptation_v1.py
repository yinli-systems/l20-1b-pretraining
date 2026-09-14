import unittest

import torch

from train_openimages_evidence_router_adaptation_v1 import (
    balanced_batches,
    balanced_pointer_ce,
    pointer_targets,
    prompt_only_rows,
    select_replay,
)


class TrainOpenImagesEvidenceRouterAdaptationTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for index in range(8):
            self.rows.extend([
                {"family_id": f"f{index}", "variant": "point", "expected_action": "POINT", "target_patch_index_14x14": index},
                {"family_id": f"f{index}", "variant": "stop", "expected_action": "STOP", "target_patch_index_14x14": None},
            ])

    def test_batches_are_balanced_and_cover_once(self):
        batches = balanced_batches(self.rows, 8, 7, 0)
        flattened = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flattened), list(range(len(self.rows))))
        for batch in batches:
            self.assertEqual(sum(self.rows[index]["expected_action"] == "POINT" for index in batch), 4)

    def test_replay_selection_keeps_complete_families(self):
        replay = []
        for index in range(8):
            action = "POINT" if index < 4 else "STOP"
            target = index if action == "POINT" else None
            replay.extend([
                {"family_id": f"r{index}", "variant": "base", "expected_action": action, "target_patch_index_14x14": target},
                {"family_id": f"r{index}", "variant": "change", "expected_action": action, "target_patch_index_14x14": target},
            ])
        selected = select_replay(replay, 4, 11)
        self.assertEqual(len(selected), 8)
        self.assertEqual(len({row["family_id"] for row in selected}), 4)
        self.assertEqual(sum(row["expected_action"] == "POINT" for row in selected), 4)
        self.assertEqual(sum(row["expected_action"] == "STOP" for row in selected), 4)

    def test_pointer_targets_and_loss(self):
        targets, point = pointer_targets(self.rows[:2], 196)
        self.assertEqual(targets.tolist(), [0, 196])
        logits = torch.zeros(2, 197)
        loss = balanced_pointer_ce(logits, targets, point)
        self.assertTrue(torch.isfinite(loss))

    def test_prompt_only_rows_do_not_mutate_sources(self):
        source = [{"question": "q", "answer": "a very long answer"}]
        transformed = prompt_only_rows(source)
        self.assertEqual(transformed[0]["answer"], "ok")
        self.assertEqual(source[0]["answer"], "a very long answer")


if __name__ == "__main__":
    unittest.main()
