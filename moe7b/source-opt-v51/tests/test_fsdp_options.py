from pathlib import Path

from cvcr_moe.train_fsdp import parser


def parse(*extra: str):
    return parser().parse_args(
        [
            "--config",
            str(Path("model.json")),
            "--data-dir",
            str(Path("train")),
            "--val-dir",
            str(Path("validation")),
            "--output-dir",
            str(Path("run")),
            "--target-tokens",
            "1000000",
            *extra,
        ]
    )


def test_fsdp_communication_options_default_to_current_numerics():
    arguments = parse()
    assert arguments.keep_unsharded_between_microbatches is False
    assert arguments.keep_root_unsharded_after_forward is False
    assert arguments.reduce_dtype == "float32"


def test_fsdp_communication_screen_options_are_explicit():
    arguments = parse(
        "--sync-every-microbatch",
        "--keep-unsharded-between-microbatches",
        "--keep-root-unsharded-after-forward",
        "--reduce-dtype",
        "bfloat16",
    )
    assert arguments.sync_every_microbatch is True
    assert arguments.keep_unsharded_between_microbatches is True
    assert arguments.keep_root_unsharded_after_forward is True
    assert arguments.reduce_dtype == "bfloat16"
