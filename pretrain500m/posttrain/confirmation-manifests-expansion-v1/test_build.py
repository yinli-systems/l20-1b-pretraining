import tempfile
from pathlib import Path
import unittest

import numpy as np

import build


class PackedArrayTests(unittest.TestCase):
    def test_npy_header_and_exact_token_shape_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'train.blocks.npy'
            np.save(path,np.zeros(2*2049,dtype=np.uint16))
            metadata={'blocks':2,'array_tokens':2*2049,'prediction_tokens':2*2048,'dtype':'uint16'}
            build.validate_packed_array(path,metadata,'fixture')
            self.assertGreater(path.stat().st_size,2*2049*2)

    def test_wrong_shape_or_dtype_fails(self):
        metadata={'blocks':2,'array_tokens':2*2049,'prediction_tokens':2*2048,'dtype':'uint16'}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'train.blocks.npy'
            np.save(path,np.zeros(2*2049-1,dtype=np.uint16))
            with self.assertRaisesRegex(ValueError,'array/block mismatch'):
                build.validate_packed_array(path,metadata,'fixture')
            np.save(path,np.zeros(2*2049,dtype=np.int32))
            with self.assertRaisesRegex(ValueError,'array/block mismatch'):
                build.validate_packed_array(path,metadata,'fixture')


if __name__=='__main__':unittest.main()
