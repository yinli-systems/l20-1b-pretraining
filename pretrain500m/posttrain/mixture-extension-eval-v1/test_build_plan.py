import pytest

from build_plan import parse_slurm_states, build


def test_requires_successful_slurm_completions_before_plan_build(tmp_path):
    states = parse_slurm_states("1591570|COMPLETED|0:0\n1591571|RUNNING|0:0\n")
    assert states[1591570] == ("COMPLETED", "0:0")
    assert states[1591571] == ("RUNNING", "0:0")
    with pytest.raises(ValueError, match="not completed successfully"):
        build(tmp_path, tmp_path / "plan", states)
