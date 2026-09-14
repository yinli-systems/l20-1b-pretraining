import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
import scan


class OldSnapshotComparison(unittest.TestCase):
    def fixture(self, root):
        path=root/'owned.parquet'
        pq.write_table(pa.table({'text':['  ＡＢＣ\nDef  ','unmatched fixture','other owned text'],
            'url':['https://www.example.test/a','https://clean.test/b','https://other.test'],
            'language':['en','en','en'],'language_score':[0.99,0.95,0.5],'int_score':[3,4,5]}),path)
        output=root/'result';output.mkdir()
        return {'path':str(path),'sha256':scan.sha(path)},output

    def test_actual_parquet_matches_and_filter_scope(self):
        with tempfile.TemporaryDirectory() as d:
            spec,out=self.fixture(Path(d))
            scan.LOOKUP={scan.normalized_hash('abc def'):[{'source_id':'owned','tranche':0,'row':0,'text_sha256':hashlib.sha256(b'abc def').hexdigest()}],scan.normalized_hash('other owned text'):[{'source_id':'owned','row':1}]}
            result=scan.scan_shard((spec,str(out)))
            self.assertEqual(result['rows'],3);self.assertEqual(result['matched_old_rows'],2)
            self.assertEqual(result['matched_old_rows_passing_initial_filters'],1)
            with gzip.open(result['matches'],'rt') as f:hits=[json.loads(x) for x in f]
            self.assertEqual([h['old_physical_row'] for h in hits],[0,2])
            self.assertNotEqual(hits[0]['old_text_sha256'],hits[0]['new_documents'][0]['text_sha256'])
            self.assertEqual(json.loads(Path(result['host_counts']).read_text()),{'clean.test':1,'www.example.test':1})

    def test_changed_old_file_fails_before_scan(self):
        with tempfile.TemporaryDirectory() as d:
            spec,out=self.fixture(Path(d));spec['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'identity mismatch'):scan.scan_shard((spec,str(out)))
            self.assertEqual(list(out.iterdir()),[])

    def test_normalization_retains_legacy_lower_semantics(self):
        self.assertEqual(scan.normalized_hash('  ＡＢＣ\nDef  '),scan.normalized_hash('abc def'))
        self.assertNotEqual(scan.normalized_hash('Straße'),scan.normalized_hash('STRASSE'))


if __name__=='__main__':unittest.main()
