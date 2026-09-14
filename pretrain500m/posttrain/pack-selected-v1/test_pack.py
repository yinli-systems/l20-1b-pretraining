import unittest
import numpy as np
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


if __name__=='__main__':unittest.main()
