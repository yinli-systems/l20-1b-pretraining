#!/usr/bin/env python3
"""CPU-only author-adapter compatibility checks, not real-checkpoint evaluation."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('hf_olmo','openlm'), required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    from lm_eval.models.huggingface import HFLM
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if args.kind == 'hf_olmo':
        from hf_olmo import OLMoConfig, OLMoForCausalLM
        config = OLMoConfig(d_model=64,n_heads=4,n_layers=2,mlp_ratio=8,
                           embedding_size=128,vocab_size=128,max_sequence_length=32,
                           layer_norm_type='rms',layer_norm_eps=1e-6,activation_type='swiglu',
                           include_bias=False,weight_tying=False,rope=True,rope_full_precision=True,
                           flash_attention=False,attention_dropout=0.0,embedding_dropout=0.0,
                           residual_dropout=0.0,init_device='cpu',eos_token_id=127,pad_token_id=1)
        config._attn_implementation = 'sdpa'
        model = OLMoForCausalLM(config,init_params=True).eval()
    else:
        from open_lm.hf import OpenLMConfig, OpenLMForCausalLM
        config = OpenLMConfig(dim=64,n_heads=4,n_layers=2,vocab_size=128,seq_len=32,
                              norm_type='gain_only_lp_layer_norm',norm_eps=1e-5,
                              ffn_type='swiglu_torch',attn_name='torch_attn',apply_qk_norm=True,
                              positional_embedding_type='rotary',weight_tying=False)
        model = OpenLMForCausalLM(config).eval()
    x = torch.arange(2,18).reshape(2,8)
    with torch.inference_mode():
        y = model(input_ids=x,use_cache=False).logits
        z = model(input_ids=x[:,:4],use_cache=False).logits
    assert y.shape == (2,8,128) and torch.isfinite(y).all()
    assert torch.allclose(y[:,:4],z,atol=1e-5,rtol=1e-5), 'prefix causality mismatch'
    directory = Path(tempfile.mkdtemp(prefix='adapter-smoke-',dir=str(args.output.parent)))
    model.save_pretrained(directory,safe_serialization=True)
    loaded, info = AutoModelForCausalLM.from_pretrained(directory,local_files_only=True,output_loading_info=True)
    assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys'], info
    with torch.inference_mode():
        roundtrip = loaded.eval()(input_ids=x,use_cache=False).logits
        bf16 = loaded.to(torch.bfloat16)(input_ids=x,use_cache=False).logits
    assert torch.allclose(y,roundtrip,atol=1e-6,rtol=1e-6)
    assert torch.isfinite(bf16).all()
    assert not torch.cuda.is_initialized()
    record = {'kind':args.kind,'status':'cpu_toy_pass','toy_only':True,'device':'cpu',
              'prefix_causal':True,'save_load_keys_exact':True,'bf16_finite':True,
              'torch':torch.__version__,'time':time.time(),'toy_artifact':str(directory),
              'limitations':'Author adapter import/forward/serialization compatibility only; no real checkpoint weights or benchmark scores validated.'}
    with args.output.open('x') as stream:
        json.dump(record,stream,indent=2)
    print(json.dumps(record),flush=True)


if __name__ == '__main__':
    main()
