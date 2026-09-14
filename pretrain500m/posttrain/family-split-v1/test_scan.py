import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import scan


class IntegrationTest(unittest.TestCase):
    def fixture(self, root):
        raw=root/'raw';raw.mkdir();bound={};feature_files=[];ex={};n=0
        for sid in ['dclm','pdf_en']:
            records=[];features=[]
            for i in range(2):
                text=f'{sid} example number {i} with enough unique tokens for the fixture'
                h=hashlib.sha256(text.encode()).hexdigest()
                reasons=['verified_supplemental_exact_span_exclusion'] if sid=='dclm' and i==0 else []
                if reasons: ex[h]=reasons
                records.append(dict(text=text,_provenance=dict(source_id=sid)))
                features.append(dict(source_id=sid,tranche=0,row=i,text_sha256=h,encoded_tokens_including_one_eos=129,
                    canonical_family='url:https://example.org/shared' if sid=='dclm' else f'url:https://pdf.org/{i}',
                    numeric_template_sha256=None,passes_filters_and_bound_exclusions=not reasons,
                    exclusion_reasons=reasons,minhash64_u32_le_hex=f'{n+1:02x}'*256));n+=1
            path=raw/(sid+'.jsonl.gz')
            with gzip.open(path,'wt') as f:
                for r in records:f.write(json.dumps(r)+'\n')
            bound[str(path)]=scan.sha(path)
            feature=root/(sid+'.features.jsonl.gz')
            with gzip.open(feature,'wt') as f:
                for r in features:f.write(json.dumps(r)+'\n')
            feature_files.append(dict(source_id=sid,tranche=0,features=str(feature),features_sha256=scan.sha(feature),rows=2))
        def save(name,data):
            path=root/name;path.write_text(json.dumps(data));return path
        raw_report=save('raw-report.json',dict(tranches=[dict(path=str(raw))]))
        exclusions=save('exclusions.json',dict(exclude_text_sha256_reasons=ex))
        report=save('features-report.json',dict(status='QUALITY_AND_FAMILY_FEATURES_COMPLETE_NOT_ADMITTED',
            raw_report_sha256=scan.sha(raw_report),exclusion_plan_sha256=scan.sha(exclusions),rows=4,
            files=feature_files,input_bindings=bound))
        design=save('design.json',dict(protocol_id='p529m-fast-start-nosynthetic-v1',development=dict(
            minimum_distinct_document_families_per_domain=1000,minimum_prediction_tokens_per_domain=1048576)))
        return argparse.Namespace(features_report=report,expected_features_sha256=scan.sha(report),raw_report=raw_report,
            exclusions=exclusions,design=design,expected_design_sha256=scan.sha(design),output=root/'out',workers=1)

    def setUp(self):
        scan.RECORDS=[];scan.LOOKUP={};scan.NEEDED=set();scan.TEXTS={};scan.cached_shingles.cache_clear()

    def test_bound_pipeline_emits_quarantine_and_explicit_deficits(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp))
            with patch.object(scan.shutil,'disk_usage',return_value=argparse.Namespace(free=100*1024**3)):
                scan.run(args)
            report=json.loads((args.output/'report.json').read_text())
            self.assertEqual(report['rows'],4);self.assertEqual(report['quarantined_rows'],2)
            self.assertEqual(report['selected_unique_eligible_documents'],2);self.assertFalse(report['training_admitted'])
            with gzip.open(args.output/'row-assignments.jsonl.gz','rt') as f: rows=[json.loads(l) for l in f]
            self.assertTrue(all(r['partition']=='excluded' for r in rows if r['source_id']=='dclm'))
            self.assertEqual(len({r['family_id'] for r in rows if r['source_id']=='dclm'}),1)
            self.assertGreater(report['reserved_partitions']['deficits']['development']['math']['families'],0)
            self.assertTrue(all(scan.sha(p)==h for p,h in report['output_files'].items()))

    def test_tampered_upstream_report_fails_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp));args.features_report.write_text('{}')
            with self.assertRaisesRegex(ValueError,'input identity mismatch'):scan.run(args)
            self.assertFalse(args.output.exists())


if __name__=='__main__':unittest.main()
