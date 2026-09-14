#!/usr/bin/env python3
from __future__ import annotations

import unittest

import torch

from counterfactual_losses import (
    answer_preference,
    candidate_distillation_loss,
    compression_induced_failure,
    evidence_delta_loss,
    masked_sequence_logprob,
    pair_joint_correct,
    paired_cluster_bootstrap,
)


class CounterfactualLossTests(unittest.TestCase):
    def test_binary_candidate_distillation_matches_teacher_distribution(self) -> None:
        teacher = torch.tensor([[[2.0, -1.0], [-0.5, 0.5]]])
        student = teacher.clone().requires_grad_(True)
        matching = candidate_distillation_loss(student, teacher, temperature=2.0)
        self.assertAlmostEqual(float(matching.detach()), 0.0, places=6)
        mismatched = candidate_distillation_loss(-student, teacher, temperature=2.0)
        self.assertGreater(float(mismatched.detach()), 0.0)
        mismatched.backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_masked_logprob_excludes_padding(self) -> None:
        logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [50.0, -50.0]]])
        tokens = torch.tensor([[0, 1, 1]])
        mask = torch.tensor([[True, True, False]])
        result = masked_sequence_logprob(logits, tokens, mask)
        expected = 2 * torch.log_softmax(torch.tensor([2.0, 0.0]), dim=-1)[0]
        self.assertTrue(torch.allclose(result[0], expected))

    def test_evidence_delta_uses_only_reliable_teacher_pairs(self) -> None:
        sb = torch.tensor([3.0, 100.0], requires_grad=True)
        se = torch.tensor([-2.0, -100.0], requires_grad=True)
        tb = torch.tensor([2.0, 0.0])
        te = torch.tensor([-2.0, 0.0])
        reliable = torch.tensor([True, False])
        loss = evidence_delta_loss(sb, se, tb, te, reliable)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertEqual(float(sb.grad[1]), 0.0)
        self.assertEqual(float(se.grad[1]), 0.0)

    def test_no_reliable_teacher_pair_is_zero_but_differentiable(self) -> None:
        sb = torch.tensor([1.0], requires_grad=True)
        loss = evidence_delta_loss(
            sb,
            torch.tensor([0.0], requires_grad=True),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            torch.tensor([False]),
        )
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(sb.grad), 0.0)

    def test_joint_correct_and_compression_failure(self) -> None:
        full_a = torch.tensor([True, True, False])
        full_b = torch.tensor([True, False, True])
        comp_a = torch.tensor([True, True, True])
        comp_b = torch.tensor([False, True, True])
        self.assertTrue(torch.equal(pair_joint_correct(full_a, full_b), torch.tensor([True, False, False])))
        failures, eligible = compression_induced_failure(full_a, full_b, comp_a, comp_b)
        self.assertTrue(torch.equal(eligible, torch.tensor([True, False, False])))
        self.assertTrue(torch.equal(failures, torch.tensor([True, False, False])))

    def test_preference_and_cluster_bootstrap_are_deterministic(self) -> None:
        self.assertTrue(
            torch.equal(
                answer_preference(torch.tensor([3.0]), torch.tensor([1.0])),
                torch.tensor([2.0]),
            )
        )
        first = paired_cluster_bootstrap(
            [1.0, 1.0, -1.0, -1.0], ["a", "a", "b", "b"], resamples=1000
        )
        second = paired_cluster_bootstrap(
            [1.0, 1.0, -1.0, -1.0], ["a", "a", "b", "b"], resamples=1000
        )
        self.assertEqual(first, second)
        self.assertEqual(first.estimate, 0.0)
        self.assertEqual(first.clusters, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
