import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pack
from pack import write_span


class PackingTest(unittest.TestCase):
    def test_documents_across_blocks_and_retained_tail(self):
        a=np.zeros(12,dtype=np.uint16);tail=[];position=0
        for document in [[1,2,3,50279],[4,5,6,7,8,50279],[9,10,11,50279]]:
            position=write_span(a,position,document,tail)
        self.assertEqual(position,14)
        self.assertEqual(a.tolist()+tail,[1,2,3,50279,4,5,6,7,8,50279,9,10,11,50279])

    def test_reserved_stream_retains_last_document(self):
        a=np.zeros(5,dtype=np.uint16);tail=[]
        self.assertEqual(write_span(a,0,[1,2,3,4,50279],tail),5)
        self.assertFalse(tail);self.assertEqual(a[-1],50279)

    def test_reject_overflow_before_write(self):
        a=np.zeros(2,dtype=np.uint16);tail=[]
        with self.assertRaises(ValueError):write_span(a,0,[1,65536],tail)
        self.assertEqual(a.tolist(),[0,0]);self.assertEqual(tail,[])

    def test_reused_source_requires_exact_current_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);array_path=root/'train-dclm.blocks.npy';np.save(array_path,np.zeros(2049,dtype=np.uint16))
            tail_path=root/'dclm.train-tail.json';tail_path.write_text(json.dumps({'token_ids':[],'virtual_offset':2049}))
            index_path=root/'dclm.documents.jsonl.gz'
            row={'source_id':'dclm','tranche':0,'row':0,'partition':'train','family_id':'family',
                 'text_sha256':'a'*64,'start':0,'end':2049,'first_document_target_offset':1,'eos_id':50279}
            with gzip.open(index_path,'wt') as f:f.write(json.dumps(row)+'\n')
            result={'status':'SELECTED_SOURCE_PACKED_NOT_ADMITTED','source_id':'dclm',
                'document_index':str(index_path),'document_index_sha256':pack.sha(index_path),
                'outputs':{'train':{'path':str(array_path),'sha256':pack.sha(array_path),'documents':1,
                    'encoded_tokens':2049,'array_tokens':2049,'tail_tokens':0,'dtype':'uint16','blocks':1,
                    'prediction_tokens':2048,'tail_path':str(tail_path),'tail_sha256':pack.sha(tail_path)}}}
            pack.ROWS={('dclm',0,0):{'partition':'train','family_id':'family','text_sha256':'a'*64,
                                      'encoded_tokens_including_one_eos':2049}}
            pack.RETENTION={'train':{'dclm':{'documents':1,'encoded_tokens':2049}}}
            self.assertTrue(pack.validate_reused_source('dclm',result))
            pack.ROWS[('dclm',0,1)]={'partition':'train','family_id':'new','text_sha256':'b'*64,
                                     'encoded_tokens_including_one_eos':2049}
            self.assertFalse(pack.validate_reused_source('dclm',result))


if __name__=='__main__':unittest.main()
