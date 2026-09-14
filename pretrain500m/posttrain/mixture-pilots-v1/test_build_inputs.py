import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name('build_inputs.py')
SPEC = importlib.util.spec_from_file_location('build_inputs', MODULE_PATH)
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class QuotaTests(unittest.TestCase):
    def test_all_recipes_have_exact_budget_and_close_proportions(self):
        for recipe, basis_points in BUILD.RECIPES_BPS.items():
            quotas = BUILD.exact_block_quotas(basis_points)
            self.assertEqual(sum(quotas.values()), BUILD.TARGET_BLOCKS, recipe)
            for source, bps in basis_points.items():
                exact = BUILD.TARGET_BLOCKS * bps / 10_000
                self.assertLessEqual(abs(quotas[source] - exact), 1, (recipe, source))

    def test_source_set_and_percentage_are_frozen(self):
        expected = set(BUILD.RECIPES_BPS['R0_F2_current_control'])
        for basis_points in BUILD.RECIPES_BPS.values():
            self.assertEqual(set(basis_points), expected)
            self.assertEqual(sum(basis_points.values()), 10_000)

    def test_capacity_accounts_for_parent_exposure(self):
        source = {
            'max_cumulative_epochs': '2.2',
            'prior_prediction_tokens': 3 * BUILD.SEQUENCE_LENGTH,
            'shards': [{'blocks': 10}],
        }
        self.assertEqual(BUILD.additional_block_capacity(source), 19)

    def test_seven_arms_cover_five_mixtures_and_r3_lr_grid(self):
        self.assertEqual(len(BUILD.ARMS), 7)
        self.assertEqual({recipe for _, recipe, _ in BUILD.ARMS}, set(BUILD.RECIPES_BPS))
        r3_lrs = {lr for _, recipe, lr in BUILD.ARMS if recipe == 'R3_current_balanced'}
        self.assertEqual(r3_lrs, {'0.00003', '0.00006', '0.0001'})


if __name__ == '__main__':
    unittest.main()
