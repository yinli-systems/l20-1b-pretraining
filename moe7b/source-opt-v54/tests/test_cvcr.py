from dataclasses import replace
from pathlib import Path

import torch
import pytest

from cvcr_moe.config import ModelConfig
from cvcr_moe.train import verify_digest_receipt
from cvcr_moe.cvcr import ProbeSampler, control_variate_estimate
from cvcr_moe.model import MoELanguageModel, invert_permutation
from cvcr_moe.train_fsdp import (
    RUN_ROLE_CLAIMS,
    checkpoint_with_cpu_saved_tensors,
    clear_bf16_adam_gradients,
    normalize_bf16_adam_states,
    parser as fsdp_parser,
    prepare_bf16_adam_gradients,
    register_bf16_adam_states,
)


def tiny_config(method: str = "top2") -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        eos_token_id=127,
        hidden_size=32,
        expert_intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        experts_per_token=2,
        max_position_embeddings=16,
        expert_backend="loop",
        method=method,
        cvcr_rank=4,
        probe_fraction=1.0,
    )


def test_fsdp_run_roles_are_explicit_and_fail_closed() -> None:
    assert set(RUN_ROLE_CLAIMS) == {
        "bounded_feasibility_screen",
        "systems_stability_pilot",
        "formal_pretraining",
    }
    parsed = fsdp_parser().parse_args(
        [
            "--config", "config.json",
            "--data-dir", "train",
            "--val-dir", "val",
            "--output-dir", "run",
            "--target-tokens", "1",
            "--sequence-length", "1024",
            "--split-expert-fsdp",
            "--split-embedding-fsdp",
            "--disable-backward-prefetch",
            "--sync-every-microbatch",
            "--run-role", "systems_stability_pilot",
        ]
    )
    assert parsed.run_role == "systems_stability_pilot"
    assert parsed.sequence_length == 1024
    assert parsed.split_expert_fsdp
    assert parsed.split_embedding_fsdp
    assert parsed.disable_backward_prefetch
    assert parsed.sync_every_microbatch


def test_checkpoint_cpu_saved_tensor_path_preserves_gradients() -> None:
    value = torch.randn(4, 8, requires_grad=True)
    reference = value.detach().clone().requires_grad_(True)

    observed_loss = checkpoint_with_cpu_saved_tensors(
        lambda tensor: tensor.sin().square().sum(), value
    )
    reference_loss = reference.sin().square().sum()
    observed_loss.backward()
    reference_loss.backward()

    torch.testing.assert_close(observed_loss, reference_loss, rtol=0.0, atol=0.0)
    torch.testing.assert_close(value.grad, reference.grad, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not hasattr(torch.empty(1), "grad_dtype"),
    reason="explicit mixed gradient dtype requires torch 2.11",
)
def test_single_tensor_bf16_adam_updates_fp32_master_and_restores_grad_contract() -> None:
    parameter = torch.nn.Parameter(torch.ones(16, dtype=torch.float32))
    model = torch.nn.ParameterList([parameter])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=False,
        foreach=False,
    )
    register_bf16_adam_states(optimizer)
    parameter.grad = torch.full_like(parameter, 0.25)
    prepare_bf16_adam_gradients(model)
    assert parameter.dtype == torch.float32
    assert parameter.grad.dtype == torch.bfloat16
    optimizer.step()
    clear_bf16_adam_gradients(model)
    assert parameter.grad is None
    assert parameter.grad_dtype == torch.float32
    assert optimizer.state[parameter]["exp_avg"].dtype == torch.bfloat16
    assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.bfloat16
    assert torch.all(parameter < 1.0)

    saved = optimizer.state_dict()
    reloaded = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=False,
        foreach=False,
    )
    reloaded.load_state_dict(saved)
    normalize_bf16_adam_states(reloaded)
    assert reloaded.state[parameter]["exp_avg"].dtype == torch.bfloat16
    assert reloaded.state[parameter]["exp_avg_sq"].dtype == torch.bfloat16


def test_inverse_permutation_scatter_matches_second_sort() -> None:
    generator = torch.Generator().manual_seed(20260916)
    for size in (0, 1, 2, 17, 4096):
        order = torch.randperm(size, generator=generator)
        torch.testing.assert_close(invert_permutation(order), order.argsort())


def test_mfu_flop_conventions_are_explicit_and_stable() -> None:
    config = ModelConfig.from_json(Path("configs/model/oracle_top2.json"))
    model = MoELanguageModel(config)
    assert model.active_flops_per_token(2048) == 628_641_792
    assert model.causal_matmul_flops_per_token(2048) == 553_181_184


def test_unknown_expert_backend_fails_closed() -> None:
    try:
        replace(tiny_config(), expert_backend="unknown")
    except ValueError as error:
        assert "unsupported expert backend" in str(error)
    else:
        raise AssertionError("unknown expert backend was accepted")


def test_control_variate_estimator_is_empirically_unbiased() -> None:
    generator = torch.Generator().manual_seed(7)
    baseline = torch.tensor([0.2, -0.5, 0.8])
    exact = torch.tensor([1.0, 0.25, -0.4])
    probability = torch.tensor([0.1, 0.25, 0.8])
    estimates = []
    for _ in range(100_000):
        included = torch.rand(3, generator=generator) < probability
        estimates.append(
            control_variate_estimate(baseline, exact, included, probability)
        )
    observed = torch.stack(estimates).mean(dim=0)
    torch.testing.assert_close(observed, exact, atol=0.02, rtol=0.0)


def test_abp_probabilities_have_positive_support() -> None:
    sampler = ProbeSampler(4, 1.0, 1e-4, "abp", 0.99)
    sampler.residual_variance.copy_(torch.tensor([1.0, 4.0, 9.0, 16.0]))
    selected = torch.tensor([[0, 1], [1, 2], [2, 3]])
    rows, expert, inclusion = sampler.sample(selected)
    assert rows.numel() == selected.shape[0]
    assert torch.all(inclusion > 0)
    assert torch.all(expert[:, None] != selected.index_select(0, rows))


def test_uniform_shared_probe_reports_exact_marginal_probability() -> None:
    torch.manual_seed(5)
    sampler = ProbeSampler(16, 1.0 / 16.0, 1e-4, "uniform", 0.99)
    first = torch.arange(1024) % 16
    selected = torch.stack((first, (first + 1) % 16), dim=-1)
    for _ in range(32):
        rows, expert, inclusion = sampler.sample(selected)
        assert rows.numel() == 64
        assert torch.all(expert[:, None] != selected.index_select(0, rows))
        torch.testing.assert_close(
            inclusion,
            torch.full_like(
                inclusion,
                (1.0 / 16.0) * (1.0 - (895.0 / 896.0) ** 64),
            ),
        )


def test_eval_path_matches_plain_top2_after_loading_deployed_weights() -> None:
    torch.manual_seed(11)
    top2 = MoELanguageModel(tiny_config("top2"))
    cvcr = MoELanguageModel(tiny_config("cvcr"))
    cvcr.load_state_dict(top2.state_dict(), strict=False)
    top2.eval()
    cvcr.eval()
    tokens = torch.randint(0, 128, (2, 12))
    torch.testing.assert_close(top2(tokens), cvcr(tokens), rtol=0.0, atol=0.0)


def test_all_ignored_targets_keep_loss_path_finite() -> None:
    torch.manual_seed(12)
    model = MoELanguageModel(tiny_config("top2")).train()
    tokens = torch.randint(0, 128, (1, 8))
    targets = torch.full_like(tokens, -100)
    total, cross_entropy, _, _ = model(
        tokens, targets, all_targets_ignored=True
    )
    assert torch.isfinite(total)
    assert cross_entropy.item() == 0.0
    (total * 0.0).backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Liger mask path requires CUDA")
def test_liger_partial_and_all_ignored_targets_are_finite_on_cuda() -> None:
    torch.manual_seed(120)
    config = replace(tiny_config("top2"), loss_backend="liger")
    model = MoELanguageModel(config).cuda().train()
    tokens = torch.randint(0, config.vocab_size, (2, 8), device="cuda")
    partial = torch.randint(0, config.vocab_size, (2, 8), device="cuda")
    partial[:, :3] = -100
    partial_total, partial_ce, _, _ = model(tokens, partial)
    assert torch.isfinite(partial_total)
    assert torch.isfinite(partial_ce)
    partial_total.backward()
    model.zero_grad(set_to_none=True)

    ignored = torch.full_like(tokens, -100)
    ignored_total, ignored_ce, _, _ = model(
        tokens, ignored, all_targets_ignored=True
    )
    assert torch.isfinite(ignored_total)
    assert ignored_ce.item() == 0.0
    (ignored_total * 0.0).backward()


def test_cvcr_does_not_perturb_deployed_initialization() -> None:
    torch.manual_seed(17)
    top2 = MoELanguageModel(tiny_config("top2"))
    torch.manual_seed(17)
    cvcr = MoELanguageModel(tiny_config("cvcr"))
    cvcr_deployed = {
        key: value
        for key, value in cvcr.state_dict().items()
        if ".credit_predictor." not in key and ".probe_sampler." not in key
    }
    top2_state = top2.state_dict()
    assert top2_state.keys() == cvcr_deployed.keys()
    for key in top2_state:
        torch.testing.assert_close(top2_state[key], cvcr_deployed[key], rtol=0.0, atol=0.0)


def test_probes_do_not_add_expert_parameter_gradients() -> None:
    torch.manual_seed(13)
    base = replace(
        tiny_config("cvcr"),
        cvcr_coefficient=0.0,
        predictor_loss_coefficient=0.0,
        probe_fraction=0.0,
    )
    no_probes = MoELanguageModel(base)
    with_probes = MoELanguageModel(replace(base, probe_fraction=1.0))
    with_probes.load_state_dict(no_probes.state_dict())
    no_probes.train()
    with_probes.train()
    tokens = torch.randint(0, 128, (2, 12))
    targets = torch.randint(0, 128, (2, 12))
    no_probes(tokens, targets)[0].backward()
    with_probes(tokens, targets)[0].backward()
    assert len(no_probes.layers) == len(with_probes.layers)
    for no_probe_layer, probe_layer in zip(no_probes.layers, with_probes.layers):
        torch.testing.assert_close(
            no_probe_layer.moe.experts.gate_up.grad,
            probe_layer.moe.experts.gate_up.grad,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            no_probe_layer.moe.experts.down.grad,
            probe_layer.moe.experts.down.grad,
            rtol=0.0,
            atol=0.0,
        )


def test_zero_credit_cvcr_preserves_all_deployed_gradients() -> None:
    torch.manual_seed(23)
    top2 = MoELanguageModel(tiny_config("top2")).train()
    cvcr = MoELanguageModel(
        replace(
            tiny_config("cvcr"),
            cvcr_coefficient=0.0,
            predictor_loss_coefficient=1e-3,
            probe_fraction=1.0,
        )
    ).train()
    cvcr.load_state_dict(top2.state_dict(), strict=False)
    tokens = torch.randint(0, 128, (2, 12))
    targets = torch.randint(0, 128, (2, 12))
    top2(tokens, targets)[0].backward()
    cvcr(tokens, targets)[0].backward()
    cvcr_parameters = dict(cvcr.named_parameters())
    for name, parameter in top2.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            cvcr_parameters[name].grad,
            rtol=0.0,
            atol=0.0,
        )


def test_inference_state_excludes_training_only_modules() -> None:
    model = MoELanguageModel(tiny_config("cvcr"))
    inference = model.inference_state_dict()
    assert inference
    assert all("credit_predictor" not in key for key in inference)
    assert all("probe_sampler" not in key for key in inference)


def test_declared_parameter_counts_match_modules() -> None:
    for method in ("top2", "cvcr"):
        config = tiny_config(method)
        model = MoELanguageModel(config)
        deployed = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if ".credit_predictor." not in name
        )
        training = sum(parameter.numel() for parameter in model.parameters())
        assert deployed == config.deployed_parameter_count()
        assert training == config.training_parameter_count()


def test_checkpoint_receipt_fails_closed(tmp_path) -> None:
    checkpoint = tmp_path / "resume.pt"
    receipt = tmp_path / "resume.sha256"
    checkpoint.write_bytes(b"checkpoint-v1")
    observed = __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
    receipt.write_text(f"{observed}  resume.pt\n")
    assert verify_digest_receipt(checkpoint, receipt) == observed

    checkpoint.write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_digest_receipt(checkpoint, receipt)


def test_model_config_rejects_token_ids_outside_vocabulary() -> None:
    with pytest.raises(ValueError, match="eos_token_id"):
        ModelConfig(vocab_size=32_000, eos_token_id=50_279)


def test_requested_7b_shape_has_exact_declared_counts() -> None:
    config = ModelConfig(
        vocab_size=32_000,
        hidden_size=2048,
        expert_intermediate_size=2816,
        num_hidden_layers=24,
        num_attention_heads=32,
        num_key_value_heads=4,
        num_experts=16,
        experts_per_token=2,
        tie_word_embeddings=False,
        eos_token_id=31_999,
        method="cvcr",
    )
    assert config.deployed_parameter_count() == 7_002_228_736
    assert config.active_parameter_count() == 1_188_923_392


def test_frozen_7b_scattermoe_config_has_exact_counts_and_numerics() -> None:
    config = ModelConfig.from_json(
        Path("configs/model/moe_7b_top2_scattermoe_liger.json")
    )
    assert config.deployed_parameter_count() == 7_077_103_616
    assert config.active_parameter_count() == 1_263_798_272
    assert config.method == "top2"
    assert config.expert_backend == "scattermoe"
    assert config.expert_projection_partitions == 1
    assert config.loss_backend == "liger"
    assert config.router_dtype == "float32"
    assert config.vocab_size == 50_280
    assert config.eos_token_id == 50_279


def test_expert_projection_split_preserves_declared_parameters_and_gradients() -> None:
    torch.manual_seed(29)
    model = MoELanguageModel(tiny_config("top2")).train()
    expert_bank = model.layers[0].moe.experts
    assert expert_bank.gate_up is expert_bank.gate_up_projection.weight
    assert expert_bank.down is expert_bank.down_projection.weight
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        model.config.training_parameter_count()
    )
    tokens = torch.randint(0, 128, (1, 8))
    targets = torch.randint(0, 128, (1, 8))
    model(tokens, targets)[0].backward()
    assert expert_bank.gate_up.grad is not None
    assert expert_bank.down.grad is not None


def test_partitioned_expert_projections_preserve_cpu_math_and_leaf_gradients() -> None:
    torch.manual_seed(31)
    reference = MoELanguageModel(tiny_config("top2")).train()
    partitioned = MoELanguageModel(
        replace(tiny_config("top2"), expert_projection_partitions=2)
    ).train()

    reference_state = reference.state_dict()
    partitioned_state = partitioned.state_dict()
    for name, value in reference_state.items():
        if ".gate_up_projection.weight" in name:
            prefix = name.replace(".gate_up_projection.weight", "")
            chunks = value.chunk(2, dim=1)
            for index, chunk in enumerate(chunks):
                partitioned_state[
                    f"{prefix}.gate_up_projection.projections.{index}.weight"
                ].copy_(chunk)
        elif ".down_projection.weight" in name:
            prefix = name.replace(".down_projection.weight", "")
            chunks = value.chunk(2, dim=1)
            for index, chunk in enumerate(chunks):
                partitioned_state[
                    f"{prefix}.down_projection.projections.{index}.weight"
                ].copy_(chunk)
        else:
            partitioned_state[name].copy_(value)

    tokens = torch.randint(0, 128, (1, 8))
    targets = torch.randint(0, 128, (1, 8))
    reference_loss = reference(tokens, targets)[0]
    partitioned_loss = partitioned(tokens, targets)[0]
    torch.testing.assert_close(reference_loss, partitioned_loss, rtol=1e-6, atol=1e-6)
    partitioned_loss.backward()
    for layer in partitioned.layers:
        units = layer.moe.experts.fsdp_projection_units()
        assert len(units) == 4
        assert all(unit.weight.grad is not None for unit in units)
    assert sum(parameter.numel() for parameter in partitioned.parameters()) == (
        partitioned.config.training_parameter_count()
    )


def test_cvcr_disabled_when_coefficient_is_zero() -> None:
    base = tiny_config("cvcr")
    config = replace(base, cvcr_coefficient=0.0, predictor_loss_coefficient=0.0)
    model = MoELanguageModel(config)
    model.train()
    tokens = torch.randint(0, 128, (1, 8))
    targets = torch.randint(0, 128, (1, 8))
    loss, *_ = model(tokens, targets)
    loss.backward()
    assert torch.isfinite(loss)


def test_temporally_skipped_cvcr_does_not_touch_predictor_gradients() -> None:
    config = replace(tiny_config("cvcr"), cvcr_update_interval=2, probe_fraction=0.25)
    model = MoELanguageModel(config).train()
    model.set_cvcr_active(False)
    tokens = torch.randint(0, 128, (1, 8))
    targets = torch.randint(0, 128, (1, 8))
    loss, *_ = model(tokens, targets)
    loss.backward()
    for name, parameter in model.named_parameters():
        if ".credit_predictor." in name:
            assert parameter.grad is None
