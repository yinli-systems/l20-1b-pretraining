import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import scan
from features import PublicSuffix


class ScannerTests(unittest.TestCase):
    def test_bound_raw_identity_and_exclusions(self):
        text='This owned fixture explains a scientific idea with complete sentences and clear reasoning.'
        h=hashlib.sha256(text.encode()).hexdigest()
        scan.PSL=PublicSuffix('com\n');scan.MODEL=SimpleNamespace(f=SimpleNamespace(predict=lambda *a:[(0.99,'__label__en')]))
        scan.INDEX={('dclm',0,0):{'text_sha256':h,'tokens':160}};scan.EXCLUSIONS={h:['owned_test_exclusion']}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/'owned.jsonl.gz'
            with gzip.open(raw,'wt') as f:f.write(json.dumps({'text':text,'url':'https://example.com/fixture','language':'en','language_score':0.99,'_provenance':{'source_id':'dclm'}})+'\n')
            result=scan.process_one(('dclm',0,str(raw),1,str(root)))
            self.assertEqual(result['rows'],1);self.assertEqual(result['passed_rows'],0)
            with gzip.open(result['features'],'rt') as f:row=json.loads(next(f))
            self.assertTrue(row['passes_declared_filters']);self.assertFalse(row['passes_filters_and_bound_exclusions'])
            self.assertEqual(row['exclusion_reasons'],['owned_test_exclusion'])

    def test_changed_text_fails(self):
        scan.INDEX={('dclm',0,0):{'text_sha256':'0'*64,'tokens':160}}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/'owned.jsonl.gz'
            with gzip.open(raw,'wt') as f:f.write(json.dumps({'text':'changed owned text','_provenance':{'source_id':'dclm'}})+'\n')
            with self.assertRaisesRegex(ValueError,'bound tokenizer index'):
                scan.process_one(('dclm',0,str(raw),1,str(root)))
            self.assertFalse((root/'0-dclm.progress.json').exists())


if __name__=='__main__':unittest.main()
