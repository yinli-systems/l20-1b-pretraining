#!/usr/bin/env python3
from __future__ import annotations

import unittest

from evaluate_binding_text_oracle import describe_scene, oracle_prompt, variant_states


class BindingTextOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "question": "Is the leftmost object red?",
            "scene_state": {
                "background": "white",
                "objects": [
                    {"id": "target", "color": "red", "shape": "circle", "x": 40, "y": 80},
                    {"id": "partner", "color": "blue", "shape": "square", "x": 210, "y": 80},
                    {"id": "control", "color": "green", "shape": "triangle", "x": 120, "y": 180},
                ],
            },
            "invariant_edit": {"after": "black"},
        }

    def test_variant_states_swap_only_target_partner_colors(self) -> None:
        states = variant_states(self.row)
        edited = {item["id"]: item for item in states["edited"]["objects"]}
        self.assertEqual(edited["target"]["color"], "blue")
        self.assertEqual(edited["partner"]["color"], "red")
        self.assertEqual(edited["control"]["color"], "green")
        self.assertEqual(states["invariant"]["background"], "black")

    def test_description_orders_objects_by_x(self) -> None:
        description = describe_scene(self.row["scene_state"])
        self.assertLess(description.index("color=red"), description.index("color=green"))
        self.assertLess(description.index("color=green"), description.index("color=blue"))

    def test_oracle_prompt_reflects_edited_value(self) -> None:
        prompt = oracle_prompt(self.row, "edited")
        self.assertIn("object 1: color=blue", prompt)
        self.assertIn(self.row["question"], prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
