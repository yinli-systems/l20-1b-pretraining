from pathlib import Path
import numpy as np
import pytest
from data import PackedReader


def test_disjoint_rank_batches_and_shift(tmp_path):
    for i in range(3):
        np.save(tmp_path/f'{i}.npy',np.arange(i*100,(i+1)*100,dtype=np.uint16))
    r=PackedReader(tmp_path,seed=3,sequence_length=9)
    used=[]
    for rank in range(4):
        x,y=r.batch_for_step(1,rank,4,2)
        assert (x[:,1:]==y[:,:-1]).all()
        used.extend(x[:,0].tolist())
    assert len(used)==len(set(used))


def test_blocks_use_2049_style_stride_and_repeat_each_epoch(tmp_path):
    # sequence_length=3 means independently packed four-token blocks.
    np.save(tmp_path/'only.npy',np.arange(24,dtype=np.uint16))
    reader=PackedReader(tmp_path,seed=0,sequence_length=3,repeat=True)
    first=[]
    second=[]
    for step in range(reader.total_sequences):
        x,_=reader.batch_for_step(step,0,1,1)
        first.append(int(x[0,0]))
        x,_=reader.batch_for_step(step+reader.total_sequences,0,1,1)
        second.append(int(x[0,0]))
    assert first==[0,4,8,12,16,20]
    assert sorted(first)==sorted(second)
    assert reader.unique_prediction_tokens==18


def test_invalid_and_exhaustion(tmp_path):
    np.save(tmp_path/'bad.npy',np.arange(20,dtype=np.int32))
    with pytest.raises(ValueError):PackedReader(tmp_path,0,4)


def test_rejects_partial_packed_block(tmp_path):
    np.save(tmp_path/'partial.npy',np.arange(11,dtype=np.uint16))
    with pytest.raises(ValueError):PackedReader(tmp_path,0,4)
