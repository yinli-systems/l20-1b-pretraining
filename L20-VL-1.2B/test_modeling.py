#!/usr/bin/env python3
from __future__ import annotations

import unittest

import torch

from modeling import (
    AdaptiveSpatialPoolCompressor,
    BindingResidual,
    CompressionSpec,
    MultimodalBridge,
    QueryConditionedReadout,
    bridge_from_architecture,
    freeze,
    load_bridge_parent,
)


class BridgeTests(unittest.TestCase):
    def test_query_readout_zero_init_preserves_parent_and_token_budget(self) -> None:
        parent = MultimodalBridge(
            input_tokens=16,
            target_ratio=4,
            vision_dim=32,
            language_dim=64,
            num_heads=4,
            compressor_kind="spatial_query",
        )
        candidate = MultimodalBridge(
            input_tokens=16,
            target_ratio=4,
            vision_dim=32,
            language_dim=64,
            num_heads=4,
            compressor_kind="spatial_query",
            query_readout_rank=8,
        )
        missing = load_bridge_parent(candidate, parent.state_dict(), allow_new_compressor=True)
        self.assertTrue(missing)
        self.assertTrue(all(name.startswith("query_readout.") for name in missing))
        text = torch.randn(2, 9, 64)
        attention = torch.ones(2, 9, dtype=torch.long)
        labels = torch.full((2, 9), -100, dtype=torch.long)
        labels[:, -2:] = 7
        vision = torch.randn(2, 16, 32)
        with torch.no_grad():
            expected = parent.inject(text, attention, labels, vision)
            inputs, mask, targets, readout_attention = candidate.inject_with_readout(
                text, attention, labels, vision
            )
        torch.testing.assert_close(inputs, expected[0])
        torch.testing.assert_close(mask, expected[1])
        torch.testing.assert_close(targets, expected[2])
        self.assertEqual(readout_attention.shape, (2, 4))
        self.assertEqual(inputs.shape[1], 9 + 4 + 2)

    def test_query_readout_cannot_see_candidate_answer_embeddings(self) -> None:
        readout = QueryConditionedReadout(language_dim=32, rank=8)
        text = torch.randn(2, 7, 32)
        changed = text.clone()
        changed[:, -2:] = torch.randn_like(changed[:, -2:]) * 100
        attention = torch.ones(2, 7, dtype=torch.long)
        labels = torch.full((2, 7), -100, dtype=torch.long)
        labels[:, -2:] = 4
        with torch.no_grad():
            before = readout.prompt_query(text, attention, labels)
            after = readout.prompt_query(changed, attention, labels)
        torch.testing.assert_close(before, after)

    def test_query_readout_separates_address_and_value_sources(self) -> None:
        readout = QueryConditionedReadout(language_dim=32, rank=8)
        with torch.no_grad():
            readout.output.weight.fill_(0.25)
        address = torch.randn(2, 4, 32)
        base_value = torch.zeros(2, 4, 32)
        edited_value = torch.ones(2, 4, 32)
        text = torch.randn(2, 7, 32)
        attention = torch.ones(2, 7, dtype=torch.long)
        labels = torch.full((2, 7), -100, dtype=torch.long)
        labels[:, -2:] = 4
        with torch.no_grad():
            base, base_attention = readout(
                address, base_value, text, attention, labels
            )
            edited, edited_attention = readout(
                address, edited_value, text, attention, labels
            )
        torch.testing.assert_close(base_attention, edited_attention)
        self.assertFalse(torch.equal(base, edited))

    def test_query_readout_attention_stays_finite_under_large_projections(self) -> None:
        readout = QueryConditionedReadout(language_dim=8, rank=4)
        with torch.no_grad():
            readout.text_key.weight.fill_(10_000)
            readout.text_value.weight.fill_(10_000)
            readout.visual_key.weight.fill_(10_000)
        visual = torch.full((2, 7, 8), 10_000.0)
        text = torch.full((2, 5, 8), 10_000.0)
        attention = torch.ones(2, 5, dtype=torch.long)
        labels = torch.full((2, 5), -100, dtype=torch.long)
        with torch.no_grad():
            _, weights = readout(visual, visual, text, attention, labels)
        self.assertTrue(torch.isfinite(weights).all())
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))

    def test_binding_residual_preserves_zero_init_parent_and_fixed_budget(self) -> None:
        parent = bridge_from_architecture({
            "compressor": "spatial_query",
            "visual_tokens": 196,
            "output_visual_tokens": 49,
        })
        candidate = bridge_from_architecture({
            "compressor": "spatial_query",
            "visual_tokens": 196,
            "output_visual_tokens": 49,
            "binding_residual_rank": 16,
        })
        missing = load_bridge_parent(candidate, parent.state_dict(), allow_new_compressor=True)
        self.assertTrue(missing)
        self.assertTrue(all(name.startswith("binding_residual.") for name in missing))
        features = torch.randn(2, 196, 768)
        with torch.no_grad():
            expected = parent.visual_tokens(features)
            actual = candidate.visual_tokens(features)
        torch.testing.assert_close(actual, expected)
        assert candidate.binding_residual is not None
        self.assertEqual(sum(p.numel() for p in candidate.binding_residual.parameters()), 143_360)
        self.assertEqual(actual.shape, (2, 49, 2048))

    def test_binding_residual_interchange_changes_only_residual_source(self) -> None:
        residual = BindingResidual(196, 49, 4, 8, rank=2)
        with torch.no_grad():
            residual.write.weight.fill_(0.25)
        base = torch.zeros(1, 196, 4)
        edited = torch.ones(1, 196, 4)
        with torch.no_grad():
            base_value = residual(base)
            edited_value = residual(edited)
        self.assertEqual(base_value.shape, (1, 49, 8))
        self.assertTrue(torch.equal(base_value, torch.zeros_like(base_value)))
        self.assertFalse(torch.equal(base_value, edited_value))

    def test_full_token_parent_can_initialize_only_new_compressor(self) -> None:
        parent = bridge_from_architecture({"compressor": "none"})
        compressed = bridge_from_architecture({"compressor": "learned_query", "target_ratio": 4})
        missing = load_bridge_parent(compressed, parent.state_dict(), allow_new_compressor=True)
        self.assertTrue(missing)
        self.assertTrue(all(name.startswith("compressor.") for name in missing))
        with self.assertRaisesRegex(RuntimeError, "bridge parent mismatch"):
            load_bridge_parent(compressed, parent.state_dict(), allow_new_compressor=False)

    def setUp(self) -> None:
        torch.manual_seed(7)
        self.text = torch.randn(2, 11, 32)
        self.attention = torch.ones(2, 11, dtype=torch.long)
        self.labels = torch.arange(22).reshape(2, 11)
        self.vision = torch.randn(2, 196, 24)

    def bridge(self, ratio: int) -> MultimodalBridge:
        return MultimodalBridge(
            input_tokens=196,
            target_ratio=ratio,
            vision_dim=24,
            language_dim=32,
            num_heads=4,
        )

    def test_ratio_accounting_is_explicit(self) -> None:
        expected = {1: 196, 4: 49, 9: 22, 16: 12}
        for ratio, tokens in expected.items():
            spec = CompressionSpec(196, ratio)
            self.assertEqual(spec.output_tokens, tokens)
            self.assertAlmostEqual(spec.achieved_ratio, 196 / tokens)

    def test_exact_output_token_accounting(self) -> None:
        for tokens in (196, 144, 100, 64, 49, 36, 25, 16, 9, 4, 1):
            spec = CompressionSpec(196, 196 / tokens, tokens)
            self.assertEqual(spec.output_tokens, tokens)
            self.assertAlmostEqual(spec.achieved_ratio, 196 / tokens)
        with self.assertRaises(ValueError):
            CompressionSpec(196, 4, 0)
        with self.assertRaises(ValueError):
            CompressionSpec(196, 4, 197)

    def test_explicit_196_uses_capacity_matched_spatial_query(self) -> None:
        bridge = MultimodalBridge(
            input_tokens=196,
            target_ratio=1,
            output_tokens=196,
            vision_dim=24,
            language_dim=32,
            num_heads=4,
            compressor_kind="spatial_query",
        )
        self.assertIsNotNone(bridge.compressor)
        self.assertEqual(bridge.compress(self.vision).shape, self.vision.shape)

    def test_architecture_accepts_exact_square_token_budget(self) -> None:
        bridge = bridge_from_architecture({
            "visual_tokens": 196,
            "compressor": "spatial_query",
            "output_visual_tokens": 100,
        })
        self.assertEqual(bridge.spec.output_tokens, 100)
        self.assertAlmostEqual(bridge.spec.achieved_ratio, 1.96)

    def test_shapes_and_visual_loss_mask(self) -> None:
        for ratio in (1, 4, 9, 16):
            bridge = self.bridge(ratio)
            inputs, attention, labels = bridge.inject(
                self.text, self.attention, self.labels, self.vision
            )
            prefix = bridge.spec.output_tokens + 2
            self.assertEqual(inputs.shape, (2, 11 + prefix, 32))
            self.assertEqual(attention.shape, (2, 11 + prefix))
            self.assertTrue(torch.equal(labels[:, :prefix], torch.full((2, prefix), -100)))
            self.assertTrue(torch.equal(labels[:, prefix:], self.labels))

    def test_text_only_path_is_exact_identity(self) -> None:
        bridge = self.bridge(4)
        result = bridge.inject(self.text, self.attention, self.labels, None)
        self.assertIs(result[0], self.text)
        self.assertIs(result[1], self.attention)
        self.assertIs(result[2], self.labels)

    def test_mask_validation_and_all_masked_rejection(self) -> None:
        bridge = self.bridge(4)
        valid = torch.ones(2, 196, dtype=torch.bool)
        valid[0, -10:] = False
        self.assertEqual(bridge.visual_tokens(self.vision, valid).shape, (2, 49, 32))
        valid[1] = False
        with self.assertRaises(ValueError):
            bridge.visual_tokens(self.vision, valid)

    def test_only_bridge_receives_gradients(self) -> None:
        parent = torch.nn.Linear(32, 17, bias=False)
        freeze(parent)
        bridge = self.bridge(9)
        inputs, _, _ = bridge.inject(self.text, self.attention, self.labels, self.vision)
        parent(inputs).float().square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in bridge.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in parent.parameters()))

    def test_parameter_free_spatial_pool_is_true_2d_baseline(self) -> None:
        compressor = AdaptiveSpatialPoolCompressor(input_tokens=196, output_tokens=49)
        output = compressor(self.vision)
        self.assertEqual(output.shape, (2, 49, 24))
        self.assertEqual(sum(parameter.numel() for parameter in compressor.parameters()), 0)
        grid = self.vision[0].transpose(0, 1).reshape(24, 14, 14)
        expected_first = grid[:, :2, :2].mean(dim=(1, 2))
        self.assertTrue(torch.allclose(output[0, 0], expected_first))

    def test_spatial_query_preserves_anchor_and_is_fully_trainable(self) -> None:
        bridge = MultimodalBridge(
            input_tokens=16,
            target_ratio=4,
            vision_dim=32,
            language_dim=64,
            num_heads=4,
            compressor_kind="spatial_query",
        )
        features = torch.randn(2, 16, 32)
        compressed = bridge.compress(features)
        pooled = bridge.compressor.pool(features)
        self.assertEqual(compressed.shape, (2, 4, 32))
        self.assertLess(float((compressed - pooled).abs().mean().detach()), 0.2)
        compressed.sum().backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in bridge.compressor.parameters())
        )

    def test_spatial_pool_bridge_rejects_non_square_output(self) -> None:
        with self.assertRaises(ValueError):
            MultimodalBridge(
                input_tokens=196,
                target_ratio=9,
                vision_dim=24,
                language_dim=32,
                num_heads=4,
                compressor_kind="spatial_pool",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
