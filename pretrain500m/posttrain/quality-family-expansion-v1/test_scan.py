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

    def test_completed_result_reuses_only_hash_bound_shard(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);features=root/'0-dclm.features.jsonl.gz'
            with gzip.open(features,'wt') as f:f.write('{}\n')
            result={'status':'COMPLETED','source_id':'dclm','tranche':0,'rows':1,
                'passed_rows':1,'passed_encoded_token_upper_bound_before_eligible_dedup_and_reserves':160,
                'flags':{},'features':str(features),'features_sha256':scan.sha(features),'elapsed_seconds':1.0}
            (root/'0-dclm.progress.json').write_text(json.dumps(result))
            item=('dclm',0,'unused.jsonl.gz',1,str(root))
            self.assertEqual(scan.completed_result(item),result)
            features.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'feature hash mismatch'):
                scan.completed_result(item)

    def test_incomplete_result_is_pending(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'0-dclm.progress.json').write_text(json.dumps({'status':'RUNNING'}))
            self.assertIsNone(scan.completed_result(('dclm',0,'unused.jsonl.gz',1,str(root))))

    def test_reuse_requires_append_only_raw_and_exclusion_lineage(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/'old';old.mkdir();raw=root/'raw.jsonl.gz';raw.write_bytes(b'raw')
            features=old/'0-dclm.features.jsonl.gz'
            with gzip.open(features,'wt') as f:f.write('{}\n')
            result={'status':'COMPLETED','source_id':'dclm','tranche':0,'rows':1,'passed_rows':1,
                'passed_encoded_token_upper_bound_before_eligible_dedup_and_reserves':160,'flags':{},
                'features':str(features),'features_sha256':scan.sha(features),'elapsed_seconds':1.0}
            (old/'0-dclm.progress.json').write_text(json.dumps(result))
            prior={'status':'QUALITY_AND_FAMILY_FEATURES_COMPLETE_NOT_ADMITTED','policy':scan.POLICY,
                'raw_report_sha256':'a'*64,'exclusion_plan_sha256':'b'*64,'rows':1,
                'input_bindings':{str(raw):'c'*64},'files':[result]}
            (old/'report.json').write_text(json.dumps(prior))
            report={'incremental_from_report_sha256':'a'*64,'input_bindings':{str(raw):'c'*64}}
            exclusions={'prior_union_sha256':'b'*64}
            work=[('dclm',0,str(raw),1,str(root/'new'))]
            reused=scan.reusable_results(work,old,scan.sha(old/'report.json'),report,exclusions)
            self.assertEqual(reused[('dclm',0)],result)
            exclusions['prior_union_sha256']='changed'
            with self.assertRaisesRegex(ValueError,'not an extension'):
                scan.reusable_results(work,old,scan.sha(old/'report.json'),report,exclusions)


if __name__=='__main__':unittest.main()
