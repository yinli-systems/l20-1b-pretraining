"""Small Llama-compatible decoder. All weights are initialized locally."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 50280
    hidden_size: int = 1536
    intermediate_size: int = 4096
    num_hidden_layers: int = 18
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    eos_token_id: int = 50279
    pad_token_id: int = 1

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.head_dim % 2 == 0

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    def parameter_count(self):
        d, f, l = self.hidden_size, self.intermediate_size, self.num_hidden_layers
        projections = 2*d*d + 2*d*self.num_key_value_heads*self.head_dim
        return self.vocab_size*d*(1 if self.tie_word_embeddings else 2) + l*(projections+3*d*f+2*d)+d

    def hf_dict(self):
        return dict(asdict(self), model_type='llama', architectures=['LlamaForCausalLM'],
                    hidden_act='silu', attention_bias=False, mlp_bias=False,
                    attention_dropout=0.0, head_dim=self.head_dim, use_cache=True,
                    bos_token_id=None, eos_token_id=self.eos_token_id,
                    pad_token_id=self.pad_token_id)


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight.to(x.dtype)


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nq, self.nkv, self.hd = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.q_proj = nn.Linear(c.hidden_size, self.nq*self.hd, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, self.nkv*self.hd, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, self.nkv*self.hd, bias=False)
        self.o_proj = nn.Linear(c.hidden_size, c.hidden_size, bias=False)

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b,s,self.nq,self.hd).transpose(1,2)
        k = self.k_proj(x).view(b,s,self.nkv,self.hd).transpose(1,2)
        v = self.v_proj(x).view(b,s,self.nkv,self.hd).transpose(1,2)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q = q*cos + rotate_half(q)*sin
        k = k*cos + rotate_half(k)*sin
        # Native GQA avoids materialized repeated K/V on supported SDPA kernels.
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=self.nq != self.nkv)
        return self.o_proj(y.transpose(1,2).contiguous().view(b,s,-1))


class MLP(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size,c.intermediate_size,bias=False)
        self.up_proj = nn.Linear(c.hidden_size,c.intermediate_size,bias=False)
        self.down_proj = nn.Linear(c.intermediate_size,c.hidden_size,bias=False)

    def forward(self,x):
        return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))


class Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.self_attn = Attention(c)
        self.mlp = MLP(c)
        self.input_layernorm = RMSNorm(c.hidden_size,c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size,c.rms_norm_eps)

    def forward(self,x,cos,sin):
        x = x + self.self_attn(self.input_layernorm(x),cos,sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Backbone(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.embed_tokens = nn.Embedding(c.vocab_size,c.hidden_size)
        self.layers = nn.ModuleList([Block(c) for _ in range(c.num_hidden_layers)])
        self.norm = RMSNorm(c.hidden_size,c.rms_norm_eps)


class LM(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.config = c
        self.model = Backbone(c)
        self.lm_head = nn.Linear(c.hidden_size,c.vocab_size,bias=False)
        if c.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        for p in self.parameters():
            if p.ndim > 1:
                nn.init.normal_(p,mean=0.0,std=c.initializer_range)
        for block in self.model.layers:
            nn.init.normal_(block.self_attn.o_proj.weight,std=c.initializer_range/math.sqrt(2*c.num_hidden_layers))
            nn.init.normal_(block.mlp.down_proj.weight,std=c.initializer_range/math.sqrt(2*c.num_hidden_layers))
        inv = 1.0/(c.rope_theta**(torch.arange(0,c.head_dim,2).float()/c.head_dim))
        freq = torch.outer(torch.arange(c.max_position_embeddings).float(),inv)
        emb = torch.cat([freq,freq],dim=-1)[None,None,:,:]
        self.register_buffer('cos',emb.cos(),persistent=False)
        self.register_buffer('sin',emb.sin(),persistent=False)

    def forward(self,input_ids,targets=None):
        s = input_ids.shape[1]
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x,self.cos[:,:,:s],self.sin[:,:,:s])
        logits = self.lm_head(self.model.norm(x))
        if targets is None:
            return logits
        # Targets already shifted by exactly one token in the data reader.
        return F.cross_entropy(logits.float().reshape(-1,self.config.vocab_size),targets.reshape(-1))

    def flops_per_token(self,seq_len):
        c = self.config
        # Dense train matmuls, output projection (even when tied), attention QK/AV.
        body = c.num_hidden_layers*(2*c.hidden_size**2+2*c.hidden_size*c.num_key_value_heads*c.head_dim+3*c.hidden_size*c.intermediate_size)
        return 6*(body+c.hidden_size*c.vocab_size)+12*c.num_hidden_layers*c.hidden_size*seq_len
