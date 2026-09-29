"""Read-only CUDA-graph executor for an immutable vision encoder.

Not a trainer, mutable-weight cache, or model-quality change. Each call returns
owned output storage; different threads/streams are refused rather than raced.
Use the same tile shape, FP32 input, precision policy and encoder as the reference.
"""
from __future__ import annotations
import threading, time
from contextlib import contextmanager
import torch

@contextmanager
def ieee_fp32():
    before = torch.get_float32_matmul_precision()
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(before)
        torch.backends.cudnn.allow_tf32 = cudnn

class FrozenVisionGraph:
    """Single-owner, inference-only execution; parameters must stay immutable.

    Ordinary tensor replacement/in-place edits invalidate the executor. Unsafe
    mutation through .data or external storage is outside the contract; a caller
    requiring arbitrary weight reloads must dispose of and rebuild the executor.
    No graph output is returned as a borrowed view of the replay buffer.
    """
    def __init__(self, model, real_example: torch.Tensor):
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        if real_example.device.type != 'cuda' or real_example.dtype != torch.float32:
            raise ValueError('real CUDA FP32 input required')
        if real_example.shape != (16,3,224,224): raise ValueError('fixed 16x3x224x224 tile batch required')
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('encoder must be frozen and in evaluation mode')
        self.model=model; self.shape=tuple(real_example.shape); self.device=real_example.device
        self.owner_thread=threading.get_ident()
        self.owner_stream=torch.cuda.current_stream(self.device).cuda_stream
        self.signature=self._signature()
        self.input=real_example.detach().contiguous().clone()
        self.graph=torch.cuda.CUDAGraph()
        started=time.perf_counter()
        with torch.cuda.device(self.device),ieee_fp32(),torch.no_grad(),torch.autocast('cuda',enabled=False):
            side=torch.cuda.Stream(device=self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(3): self.output=model(pixel_values=self.input).last_hidden_state
            torch.cuda.current_stream(self.device).wait_stream(side)
            torch.cuda.synchronize(self.device)
            with torch.cuda.graph(self.graph):self.output=model(pixel_values=self.input).last_hidden_state
            torch.cuda.synchronize(self.device)
        self.capture_seconds=time.perf_counter()-started
        self.calls=0

    def _signature(self):
        # Inference tensors lack a version counter; reject rather than assume safety.
        tensors=list(self.model.named_parameters())+list(self.model.named_buffers())
        return tuple((n,id(t),t.data_ptr(),t._version,tuple(t.shape),t.dtype,t.device,t.requires_grad) for n,t in tensors)

    def __call__(self, pixels:torch.Tensor)->torch.Tensor:
        if threading.get_ident()!=self.owner_thread: raise RuntimeError('different owner thread')
        if pixels.device!=self.device or pixels.dtype!=torch.float32 or tuple(pixels.shape)!=self.shape:
            raise ValueError('graph input shape/dtype/device mismatch')
        if torch.cuda.current_stream(self.device).cuda_stream!=self.owner_stream:
            raise RuntimeError('different CUDA stream; replay not safe')
        if self.model.training or self._signature()!=self.signature:
            raise RuntimeError('encoder changed; graph must be rebuilt')
        if pixels.requires_grad: raise ValueError('frozen encoder input gradient unsupported')
        with torch.no_grad():
            self.input.copy_(pixels)
            self.graph.replay()
            result=self.output.clone()
        self.calls+=1
        return result

class H896GraphFeatures:
    """Preserve the original raster packing and live projection gradient path."""
    def __init__(self,lab,spatial_module,real_example):
        self.lab=lab;self.spatial=spatial_module
        self.encoder=FrozenVisionGraph(lab.high_vision,real_example)

    def __call__(self,batch):
        if batch.get('route')!='H': raise ValueError('H896 only; other routes must stay unchanged')
        with ieee_fp32():
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                tiles=self.spatial.native_tiles(batch['pixels']).float()
                features=torch.cat([self.encoder(tiles[i:i+16]) for i in range(0,len(tiles),16)])
                packed=self.spatial.pack_features(features,len(batch['ids']))
            # A later authorized trainer may optimize this projection. It is not
            # captured, cached, detached or updated by this executor.
            with torch.autocast('cuda',enabled=False):return self.lab.bridge.highres_pack(packed)
