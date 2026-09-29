"""CPU regression tests; no benchmark relabelling, training or GPU usage."""
import hashlib,itertools,json,sqlite3,unittest
from pathlib import Path
from reconcile_data import reconcile,numeric_label,verify_rows,digest,SOURCE,ROOT,SOURCE_SHA

class SplitTests(unittest.TestCase):
    def test_symmetric(self):
        for a,b in itertools.product(('train','development','confirmation','excluded'),repeat=2):self.assertEqual(reconcile(a,b),reconcile(b,a))
    def test_never_promote_holdout(self):
        for a,b in itertools.product(('train','development','confirmation','excluded'),repeat=2):
            if a!='train' or b!='train':self.assertNotEqual(reconcile(a,b),'train')
    def test_confirmation_development_conflict(self):self.assertEqual(reconcile('confirmation','development'),'excluded')
    def test_unknown_split(self):
        with self.assertRaises(ValueError):reconcile('validation','train')
    def test_group_merge_associative(self):
        for a,b,c in itertools.product(('train','development','confirmation','excluded'),repeat=3):self.assertEqual(reconcile(reconcile(a,b),c),reconcile(a,reconcile(b,c)))
    def test_train_preserved(self):self.assertEqual(reconcile('train','train'),'train')
    def test_exclusion_absorbing(self):
        for s in ('train','development','confirmation','excluded'):self.assertEqual(reconcile(s,'excluded'),'excluded')

class LabelTests(unittest.TestCase):
    def test_scientific(self):self.assertEqual(numeric_label('-5.39e+06.'),'-5.39e+06')
    def test_scientific_negative_exp(self):self.assertEqual(numeric_label('2.3E-4.'),'2.3E-4')
    def test_numeric(self):self.assertEqual(numeric_label('3.'),'3')
    def test_decimal(self):self.assertEqual(numeric_label('3.14.'),'3.14')
    def test_valid_decimal_preserved(self):self.assertEqual(numeric_label('3.14'),'3.14')
    def test_percent(self):self.assertEqual(numeric_label('1.25%.'),'1.25%')
    def test_text_not_silently_rewritten(self):self.assertEqual(numeric_label('Answer: 3.'),'Answer: 3.')
    def test_nonfinite_not_coerced(self):self.assertEqual(numeric_label('NaN.'),'NaN.')
    def test_multi_answer_not_coerced(self):self.assertEqual(numeric_label('3. 4.'),'3. 4.')
    def test_idempotent(self):
        for s in ('3.','3.14.','5e+8.','1.5%.','-7e-2','text.'):self.assertEqual(numeric_label(numeric_label(s)),numeric_label(s))

class ActualCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=json.loads((SOURCE/'rows.json').read_text());cls.fixed=json.loads((ROOT/'prepared-v2/rows.json').read_text())
    def test_source_unchanged(self):self.assertEqual(digest(SOURCE/'rows.json'),SOURCE_SHA)
    def test_all_rows_preserved(self):self.assertEqual(len(self.fixed),9386)
    def test_actual_ledger_matches(self):
        with sqlite3.connect('file:'+str(ROOT/'prepared-v2/identity.sqlite')+'?mode=ro',uri=True) as db:self.assertEqual(verify_rows(self.fixed,self.old,db)['ledger_split_mismatches'],0)
    def test_tampered_holdout_detected(self):
        fixed=[dict(r) for r in self.fixed];row=next(r for r in fixed if r['original_pilot_split']=='confirmation');row['response']='tampered'
        with sqlite3.connect('file:'+str(ROOT/'prepared-v2/identity.sqlite')+'?mode=ro',uri=True) as db:
            with self.assertRaisesRegex(ValueError,'heldout label'):verify_rows(fixed,self.old,db)
    def test_tampered_split_detected(self):
        fixed=[dict(r) for r in self.fixed];row=next(r for r in fixed if r['original_pilot_split']=='confirmation');row['split']='train'
        with sqlite3.connect('file:'+str(ROOT/'prepared-v2/identity.sqlite')+'?mode=ro',uri=True) as db:
            with self.assertRaises(ValueError):verify_rows(fixed,self.old,db)
    def test_receipted_files_match(self):
        adm=json.loads((ROOT/'prepared-v2/admission.json').read_text())
        for name,want in adm['files'].items():self.assertEqual(digest(ROOT/'prepared-v2'/name),want)
    def test_heldout_formats_not_changed(self):
        before={r['selection_sha256']:r for r in self.old}
        for r in self.fixed:
            if r['split'] in ('development','confirmation'):self.assertEqual(r['response'],before[r['selection_sha256']]['response'])
    def test_no_reserved_train_overlap(self):
        reserved={r['image_sha256'] for r in self.fixed if r['original_pilot_split']!='train' or r['previous_ledger_split']!='train'}
        self.assertFalse(reserved&{r['image_sha256'] for r in self.fixed if r['split']=='train'})
    def test_all_train_numerics_normalized(self):
        for r in self.fixed:
            if r['split']=='train':self.assertEqual(numeric_label(r['response']),r['response'])

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    report={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'pass':result.wasSuccessful(),'scope':'CPU unit and complete actual-pilot metadata integration checks'}
    (ROOT/'data-contract-tests.json').write_text(json.dumps(report,indent=2));raise SystemExit(not result.wasSuccessful())
