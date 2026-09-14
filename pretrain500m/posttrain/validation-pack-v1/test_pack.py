import numpy as np
import pytest

from pack import document_blocks, padding_blocks


def test_short_document_masks_only_real_targets():
    blocks=list(document_blocks(np.array([7,8,50279],dtype=np.uint16),sequence_length=4,pad=50279))
    assert len(blocks)==1
    tokens,mask=blocks[0]
    assert tokens.tolist()==[7,8,50279,50279,50279]
    assert mask.tolist()==[True,True,False,False]


def test_long_document_chunks_without_cross_document_loss():
    values=np.arange(10,dtype=np.uint16)
    blocks=list(document_blocks(values,sequence_length=4,pad=99))
    assert [int(mask.sum()) for _,mask in blocks]==[4,4,1]
    assert blocks[0][0].tolist()==[0,1,2,3,4]
    assert blocks[1][0].tolist()==[4,5,6,7,8]
    assert blocks[2][0].tolist()==[8,9,99,99,99]


def test_document_requires_context_and_target():
    with pytest.raises(ValueError):list(document_blocks(np.array([50279],dtype=np.uint16),sequence_length=4))


def test_padding_only_blocks_complete_the_frozen_global_batch():
    assert padding_blocks(7819,16)==5
    assert padding_blocks(32,16)==0
