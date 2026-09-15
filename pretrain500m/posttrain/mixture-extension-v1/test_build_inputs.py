from build_inputs import TARGET_BLOCKS, remaining_capacity, scaled_quotas


def test_scaled_quotas_are_exact_and_capacity_is_lineage_bounded():
    quotas = scaled_quotas({"a": 100_000, "b": 162_144})
    assert quotas == {"a": 200_000, "b": 324_288}
    assert sum(quotas.values()) == TARGET_BLOCKS
    source = {
        "max_cumulative_epochs": 1.5,
        "prior_prediction_tokens": 2048 * 100,
        "shards": [{"blocks": 1000}, {"blocks": 1000}],
    }
    assert remaining_capacity(source) == 2900
