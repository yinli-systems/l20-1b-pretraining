import copy
import tempfile
import torch
from model import Config,LM


def toy():
    return Config(vocab_size=128,hidden_size=64,intermediate_size=160,
                  num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                  max_position_embeddings=32,eos_token_id=127)


def test_parameter_count():
    for tied in [True,False]:
        c=toy();c.tie_word_embeddings=tied
        m=LM(c)
        assert sum(p.numel() for p in m.parameters())==c.parameter_count()


def test_causal_mask_and_shift():
    torch.manual_seed(7); m=LM(toy()).eval()
    x=torch.randint(0,128,(2,17));y=x.clone();y[:,9:]=(y[:,9:]+17)%128
    a,b=m(x),m(y)
    torch.testing.assert_close(a[:,:9],b[:,:9],atol=0,rtol=0)
    torch.testing.assert_close(m(x[:,:-1],x[:,1:]),
        torch.nn.functional.cross_entropy(m(x[:,:-1]).reshape(-1,128),x[:,1:].reshape(-1)))


def test_huggingface_export_parity():
    from transformers import LlamaConfig,LlamaForCausalLM
    torch.manual_seed(11);c=toy();m=LM(c).eval()
    hc=LlamaConfig(**c.hf_dict());hc._attn_implementation='eager'
    hf=LlamaForCausalLM(hc).eval()
    hf.load_state_dict(m.state_dict(),strict=True)
    x=torch.randint(0,128,(2,23))
    torch.testing.assert_close(m(x),hf(x).logits,atol=2e-6,rtol=2e-5)


def test_learning_and_optimizer_rng_resume():
    torch.set_num_threads(2);torch.manual_seed(42)
    c=toy();m=LM(c);o=torch.optim.AdamW(m.parameters(),lr=0.006)
    x=torch.arange(24)[None,:].repeat(2,1);y=(x+1)%128
    losses=[]
    for i in range(40):
        o.zero_grad();loss=m(x,y);loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters())
        o.step();losses.append(loss.item())
    assert losses[-1]<losses[0]*0.25
    state=copy.deepcopy({'model':m.state_dict(),'optimizer':o.state_dict(),'rng':torch.get_rng_state()})
    def step(model,opt):
        xb=torch.randint(0,128,(2,12));yb=torch.randint(0,128,(2,12))
        opt.zero_grad();loss=model(xb,yb);loss.backward();opt.step();return loss
    expected=step(m,o)
    n=LM(c);p=torch.optim.AdamW(n.parameters(),lr=0.006)
    n.load_state_dict(state['model']);p.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng'])
    actual=step(n,p)
    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
    for a,b in zip(m.parameters(),n.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
