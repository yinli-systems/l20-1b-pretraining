"""Serialization/resume regression on a toy model, not target-model warmup."""
import pytest


def test_gpu_checkpoint_atomic_roundtrip(tmp_path,monkeypatch):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('GPU serialization contract runs on the target host')
    import run_continuation as rc
    from continuation_common import sha256
    import json

    monkeypatch.setattr(rc,'RUN',tmp_path)
    torch.manual_seed(123)
    model = torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.GELU(),torch.nn.Linear(16,4)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=4e-5,betas=(.9,.95),fused=True)
    x = torch.arange(32,dtype=torch.float32,device='cuda').reshape(4,8)/32
    y = torch.ones((4,4),device='cuda')
    def step(m,o):
        o.zero_grad(set_to_none=True)
        (m(x)-y).square().mean().backward()
        o.step()
    step(model,optimizer)
    directory = tmp_path/'pilot-A'
    directory.mkdir()
    path = directory/'resume.pth'
    parent = {'sha256':'test-parent'}
    rc.save_resume(path,model,optimizer,1,parent,'test-plan',123)
    receipt = json.loads(path.with_suffix('.json').read_text())
    assert receipt['sha256'] == sha256(path)
    state = torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    restored = torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.GELU(),torch.nn.Linear(16,4)).cuda()
    restored.load_state_dict(state['model'])
    other = torch.optim.AdamW(restored.parameters(),lr=4e-5,betas=(.9,.95),fused=True)
    other.load_state_dict(state['optimizer'])
    assert state['branch_step'] == 1 and state['experiment_charged_tokens_at_save'] == 123
    step(model,optimizer)
    step(restored,other)
    for a,b in zip(model.parameters(),restored.parameters()):
        assert torch.equal(a,b)
    rc.save_resume(path,restored,other,2,parent,'test-plan',456)
    assert json.loads(path.with_suffix('.json').read_text())['step'] == 2
    assert not path.with_suffix('.pth.tmp').exists()
    # An interrupted temporary file is never silently overwritten.
    path.with_suffix('.pth.tmp').touch()
    with pytest.raises(RuntimeError,match='uncommitted checkpoint temp'):
        rc.save_resume(path,restored,other,3,parent,'test-plan',789)
