from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys

import pytest

import prepare_continuation_b as pb


def test_two_checkpoint_reserve_is_explicit(monkeypatch,tmp_path):
    assert pb.MIN_FREE_BYTES == 2*13_200_800_000 + 4*2**30
    monkeypatch.setattr(pb.shutil,'disk_usage',lambda _:SimpleNamespace(free=pb.MIN_FREE_BYTES))
    pb.require_space(tmp_path)
    monkeypatch.setattr(pb.shutil,'disk_usage',lambda _:SimpleNamespace(free=pb.MIN_FREE_BYTES-1))
    with pytest.raises(RuntimeError,match='atomic-checkpoint reserve'):
        pb.require_space(tmp_path)


def test_b_quotas_do_not_change_a_sources():
    q = pb.quotas(pb.MIXES['B'],pb.GLOBAL*pb.PILOT_STEPS)
    assert q['fresh_narrative']==q['fresh_math']==q['fresh_code']==4845
    assert q['fresh_dclm']==33915
    assert q['fresh_web']==19380
    assert sum(q.values())==96900
    assert all('b-candidate' not in str(p) for p in pb.BASES.values())
    assert pb.BROOT.name=='b-candidate-v1'


def test_help_is_read_only_and_successful():
    result = subprocess.run([sys.executable,str(Path(pb.__file__)),'--help'],capture_output=True,text=True)
    assert result.returncode==0
    assert '--sources' in result.stdout
    assert 'Traceback' not in result.stderr
