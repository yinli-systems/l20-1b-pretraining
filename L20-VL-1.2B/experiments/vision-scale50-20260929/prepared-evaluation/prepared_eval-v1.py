"""Scoped reuse for a fixed-model true/wrong-image/NLL evaluation bundle.

No optimizer, training, global cache, checkpoint write or prediction relabelling.
Decode/preprocess and the frozen image path run once per microbatch. Text masks,
image ordering, precision, generation and score semantics stay unchanged.
"""
from __future__ import annotations
import threading
import torch

class PreparedEvaluation:
    def __init__(self, lab, rows, core):
        if len(rows) < 2:
            raise ValueError('Wrong-image control needs at least two images')
        identities = [r.get('image_sha256', r.get('image_cluster')) for r in rows]
        if None in identities or len(set(identities)) != len(rows):
            raise ValueError('Unique explicit image identities required')
        self.lab=lab;self.rows=rows;self.core=core;self.closed=False
        self.owner=threading.get_ident();self.stream=torch.cuda.current_stream().cuda_stream
        self.modules=[lab.language,lab.bridge,lab.vision,lab.high_vision]
        self.states=[(m,m.training) for root in self.modules for m in root.modules()]
        self.gc=lab.gc_enabled
        self.signature=self._signature()
        try:
            lab.language.eval();lab.bridge.eval();lab.set_gc(False)
            self.prompt=lab.batch(rows,prompt_only=True)
            with torch.no_grad():self.features=lab.features(self.prompt).detach()
            ids,mask,labels=core.pack_text(rows,lab.tok.pad_token_id,prompt_only=False)
            core.check_context(ids.shape[1],self.prompt['resolution'])
            self.answer={**self.prompt,'ids':ids.cuda(),'mask':mask.cuda(),'labels':labels.cuda()}
            self.pixel_shape=tuple(self.prompt['pixels'].shape)
            # No process-wide image cache. Release the pixel tensor once encoded.
            self.prompt.pop('pixels');self.answer.pop('pixels')
        except BaseException:
            self.close();raise
    def _signature(self):
        return tuple((id(p),p.data_ptr(),p._version) for module in self.modules for p in module.parameters())
    def _check(self):
        if self.closed:raise RuntimeError('Evaluation bundle already closed')
        if threading.get_ident()!=self.owner:raise RuntimeError('Evaluation bundle belongs to another thread')
        if torch.cuda.current_stream().cuda_stream!=self.stream:raise RuntimeError('Evaluation bundle belongs to another CUDA stream')
        if self._signature()!=self.signature:raise RuntimeError('Model weights changed during fixed-model evaluation')
    def _inputs(self,prompt_only,mode):
        self._check();b=self.prompt if prompt_only else self.answer
        if mode not in ('true','shuffled','none'):raise ValueError('Unknown image mode')
        with torch.no_grad():text=self.lab.language.get_input_embeddings()(b['ids'])
        features=None if mode=='none' else self.features.roll(1,0) if mode=='shuffled' else self.features
        with torch.autocast('cuda',dtype=torch.bfloat16):
            x,mask,y=self.lab.bridge.inject(text,b['mask'],b['labels'],features)
        return x.to(torch.bfloat16),mask,y
    def generate(self, max_new=32, mode='true'):
        if not 1<=max_new<=128:raise ValueError('Generation bound required')
        for row in self.rows:self.core.check_context(len(row['_pi']),row.get('_resolution',224),max_new)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            x,m,_=self._inputs(True,mode)
            ids=self.lab.language.generate(inputs_embeds=x,attention_mask=m,max_new_tokens=max_new,
                do_sample=False,use_cache=True,pad_token_id=self.lab.tok.pad_token_id,eos_token_id=self.lab.tok.eos_token_id)
        return ids,self.lab.tok.batch_decode(ids,skip_special_tokens=True)
    def diagnostics(self):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            x,m,y=self._inputs(False,'true')
            h=self.lab.base.model(inputs_embeds=x,attention_mask=m,use_cache=False).last_hidden_state
            ys=y[:,1:];ok=ys.ne(-100);logits=self.lab.base.lm_head(h[:,:-1][ok]).float()
            ls=torch.nn.functional.cross_entropy(logits,ys[ok],reduction='none')
            content=ys[ok].ne(self.lab.tok.eos_token_id)
            owner=torch.arange(len(y),device=y.device)[:,None].expand_as(ys)[ok]
            first=torch.tensor([int(torch.nonzero(owner==i)[0]) for i in range(len(y))],device=y.device)
            return {'content_nll':float(ls[content].mean()),'eos_nll':float(ls[~content].mean()),
                'first_content_accuracy':float((logits[first].argmax(-1)==ys[ok][first]).float().mean()),
                'content_token_count':int(content.sum()),'eos_token_count':int((~content).sum())}
    def close(self):
        if not self.closed:
            self.closed=True
            self.features=None;self.prompt=None;self.answer=None
            self.lab.set_gc(self.gc)
            for module,state in self.states:module.training=state
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
