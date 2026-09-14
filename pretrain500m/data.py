"""Deterministic disjoint packed-token reader for DDP and exact resume."""
import bisect
from pathlib import Path
import random
import numpy as np
import torch


class PackedReader:
    def __init__(self, directory: Path, seed: int, sequence_length: int = 2048, repeat: bool = False):
        self.sequence_length=sequence_length
        self.block_size=sequence_length+1
        self.seed=seed
        self.repeat=repeat
        paths=sorted(directory.glob('*.npy'))
        if not paths:raise FileNotFoundError(directory)
        random.Random(seed).shuffle(paths)
        self.arrays=[];self.counts=[]
        for path in paths:
            array=np.load(path,mmap_mode='r',allow_pickle=False)
            if array.dtype!=np.uint16 or array.ndim!=1:
                raise ValueError(f'invalid packed shard {path}: {array.dtype}/{array.shape}')
            if array.size%self.block_size:
                raise ValueError(f'invalid packed shard {path}: {array.size} tokens is not divisible by {self.block_size}')
            count=array.size//self.block_size
            if not count:continue
            self.arrays.append(array);self.counts.append(count)
        self.ends=np.cumsum(self.counts).tolist()
        self.total_sequences=sum(self.counts)
        self.unique_prediction_tokens=self.total_sequences*self.sequence_length

    def _physical_index(self,index):
        """Map a logical sequence to an epoch-specific sequential traversal.

        Each epoch visits every sequence exactly once.  A seeded cyclic offset
        changes the epoch boundary without turning SSD reads into random I/O.
        """
        if index<0:raise IndexError(index)
        epoch,position=divmod(index,self.total_sequences)
        if epoch and not self.repeat:raise IndexError(index)
        offset=(self.seed+epoch*0x9E3779B1)%self.total_sequences
        return (position+offset)%self.total_sequences

    def sequence(self,index):
        if not 0<=index<self.total_sequences:raise IndexError(index)
        shard=bisect.bisect_right(self.ends,index)
        prior=0 if shard==0 else self.ends[shard-1]
        offset=(index-prior)*self.block_size
        values=np.asarray(self.arrays[shard][offset:offset+self.block_size],dtype=np.int64)
        return torch.from_numpy(values.copy())

    def batch_for_step(self,step,rank,world_size,sequences_per_rank):
        base=step*world_size*sequences_per_rank+rank*sequences_per_rank
        if not self.repeat and base+sequences_per_rank>self.total_sequences:
            raise StopIteration
        values=torch.stack([self.sequence(self._physical_index(base+i)) for i in range(sequences_per_rank)])
        return values[:,:-1],values[:,1:]
