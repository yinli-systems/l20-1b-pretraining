import unittest

from evaluate_openimages_evidence_router_adaptation_v1 import (
    localization_summary,
    passes_gate,
    select_result,
    synthetic_summary,
)


def row(action, target=None, bbox=None):
    return {
        "expected_action": action,
        "target_patch_index_14x14": target,
        "source_bbox_xmin_xmax_ymin_ymax": bbox,
    }


class OpenImagesRouterAdaptationEvaluationTests(unittest.TestCase):
    def test_natural_localization_counts_stop_and_region(self):
        rows = [row("POINT", 15, [0.0, 0.2, 0.0, 0.2]), row("STOP")]
        result = localization_summary(rows, [15, 196])
        self.assertEqual(result["action_accuracy"]["estimate"], 1.0)
        self.assertEqual(result["inside_bbox"]["estimate"], 1.0)
        self.assertEqual(result["stop_accuracy"]["estimate"], 1.0)

    def test_stop_is_localization_failure(self):
        rows = [row("POINT", 0, [0.0, 0.1, 0.0, 0.1]), row("STOP")]
        result = localization_summary(rows, [196, 196])
        self.assertEqual(result["point_recall"]["estimate"], 0.0)
        self.assertEqual(result["inside_bbox"]["estimate"], 0.0)

    def test_synthetic_metrics(self):
        rows = [row("POINT", 15), row("POINT", 20), row("STOP"), row("STOP")]
        result = synthetic_summary(rows, [15, 21, 196, 0])
        self.assertEqual(result["exact_patch"]["estimate"], 0.5)
        self.assertEqual(result["within_one_patch"]["estimate"], 1.0)
        self.assertEqual(result["stop_accuracy"]["estimate"], 0.5)

    def test_gate_and_selection_choose_earliest_tie(self):
        metric = lambda value: {"estimate": value}
        gate = {
            "natural_minimum_action_accuracy": 0.9,
            "natural_minimum_point_recall": 0.9,
            "natural_minimum_stop_accuracy": 0.9,
            "natural_minimum_inside_bbox": 0.8,
            "natural_minimum_within_one_center": 0.7,
            "synthetic_minimum_exact_patch": 0.9,
            "synthetic_minimum_within_one_patch": 0.9,
            "synthetic_minimum_stop_accuracy": 0.9,
        }
        base = {
            "natural": {key: metric(1.0) for key in (
                "action_accuracy", "point_recall", "stop_accuracy", "inside_bbox", "within_one_center"
            )},
            "synthetic_retention": {key: metric(1.0) for key in (
                "exact_patch", "within_one_patch", "stop_accuracy"
            )},
        }
        self.assertTrue(passes_gate(base, gate))
        late = {**base, "name": "late", "step": 20, "gate_passed": True}
        early = {**base, "name": "early", "step": 10, "gate_passed": True}
        self.assertEqual(select_result([late, early])["name"], "early")


if __name__ == "__main__":
    unittest.main()
