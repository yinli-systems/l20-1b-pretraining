"""CPU contracts for this long run, not a mock model or accuracy benchmark."""
import collections,hashlib,json,math,unittest
from pathlib import Path
from transformers import AutoTokenizer
import stream_sources as s

class DataRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        info=json.loads((s.PROJECT/'vlm-opt-20260925T1537Z/inputs.json').read_text())
        cls.tok=AutoTokenizer.from_pretrained(info['language'],local_files_only=True)
    def choose(self,turns,domain='plotqa'):
        return s.choices(domain,turns,'0'*64,self.tok,collections.Counter())
    def test_integer(self):self.assertEqual(s.normalized_training_answer('3.'),'3')
    def test_scientific(self):self.assertEqual(s.normalized_training_answer('-5.39e+06.'),'-5.39e+06')
    def test_percent(self):self.assertEqual(s.normalized_training_answer('4.5%.'),'4.5%')
    def test_normal_decimal(self):self.assertEqual(s.normalized_training_answer('3.14'),'3.14')
    def test_prose_not_rewritten(self):self.assertEqual(s.normalized_training_answer('The value is 3.'),'The value is 3.')
    def test_nonfinite_not_numeric(self):self.assertEqual(s.normalized_training_answer('NaN.'),'NaN.')
    def test_empty_rejected(self):self.assertFalse(self.choose([{'user':'','assistant':'3'}]))
    def test_numbered_placeholder_rejected(self):self.assertFalse(self.choose([{'user':'Q1:','assistant':'Text'}]))
    def test_identifier_question_rejected(self):self.assertFalse(self.choose([{'user':'What is the bank account number?','assistant':'12345'}]))
    def test_document_spam_rejected(self):self.assertFalse(self.choose([{'user':'What is this?','assistant':'Get a free pdf download from this page.'}],'docmatix'))
    def test_contradictory_answers_rejected(self):self.assertFalse(self.choose([{'user':'How many?','assistant':'3'},{'user':'How many?','assistant':'4'}]))
    def test_target_never_truncated(self):self.assertFalse(self.choose([{'user':'What is shown?','assistant':' number'*1000}]))
    def test_original_reference_kept(self):
        row=self.choose([{'user':'How much?','assistant':'-5.3e+06.'}],domain='docmatix')[0]
        self.assertEqual(row[3],'-5.3e+06');self.assertEqual(row[4],'-5.3e+06.')
    def test_scientific_chart_labels_quarantined(self):self.assertFalse(self.choose([{'user':'What is the total trade?','assistant':'3.38e+1.'}]))
    def test_generic_download_boilerplate_rejected(self):self.assertFalse(self.choose([{'user':'Where is the guide?','assistant':'The guide is available in Book Servers with less latency.'}],'docmatix'))
    def test_selection_deterministic(self):
        rows=[{'user':'What is the value?','assistant':'3'},{'user':'How many bars?','assistant':'4'}]
        self.assertEqual(self.choose(rows),self.choose(rows))
    def test_no_hard_quality_rating_filter(self):
        self.assertTrue(self.choose([{'user':'What is the value?','assistant':'3','quality':0}]))
    def test_document_split_deterministic(self):
        doc=hashlib.sha256(b'an existing document').hexdigest()
        self.assertEqual(s.ImageLedger.split(doc),s.ImageLedger.split(doc))

class BudgetAndSourceContracts(unittest.TestCase):
    def test_real_unique_budget(self):self.assertEqual(s.CONFIG['max_new_unique_images'],s.CONFIG['max_updates']*32)
    def test_source_mixture(self):self.assertEqual(sum(s.CONFIG['new_per_step'].values()),32)
    def test_single_6h_stage(self):self.assertEqual(s.CONFIG['maximum_wall_seconds'],21600)
    def test_full_campaign_not_claimed(self):self.assertTrue(s.CONFIG['not_8_4M_completed']);self.assertEqual(s.CONFIG['campaign_unique_image_goal'],8400000)
    def test_old_anchor_unchanged(self):self.assertEqual(s.CONFIG['old_QA_anchor'],.5625);self.assertEqual(s.CONFIG['old_QA_floor'],.5125)
    def test_previous_seen_shards_not_reused(self):
        paths=[x['rfilename'] for xs in s.CONFIG['source_plan'].values() for x in xs]
        self.assertEqual(len(paths),len(set(paths)))
        self.assertFalse(any('train-00000-' in x or 'train-00001-of-01106' in x or 'train-00002-of-01106' in x for x in paths))
    def test_every_whole_source_pinned(self):
        for xs in s.CONFIG['source_plan'].values():
            for x in xs:self.assertEqual(len(x['lfs']['sha256']),64);self.assertEqual(x['size'],x['lfs']['size']);self.assertEqual(len(x['revision']),40)
    def test_download_budget_sufficient_but_bounded(self):self.assertLessEqual(s.CONFIG['source_declared_bytes'],s.CONFIG['maximum_source_download_bytes']);self.assertLess(s.CONFIG['maximum_source_download_bytes'],50*2**30)
    def test_losses_total_one(self):
        self.assertAlmostEqual(sum(s.CONFIG[k] for k in ('new_loss_weight','old_QA_CE_weight','old_caption_CE_weight','old_domain_replay_CE_weight')),1.)
    def test_original_models_preserved(self):self.assertTrue(s.CONFIG['vision_and_original_language_frozen']);self.assertFalse(s.CONFIG['automatic_model_promotion'])
    def test_checkpoint_anchors_exist_and_match(self):
        for label in ('warmstart','teacher'):
            self.assertEqual(s.sha(Path(s.CONFIG[label+'_checkpoint'])/'manifest.json'),s.CONFIG[label+'_manifest_sha256'])
    def test_genuine_optimizer_state(self):
        manifest=json.loads((Path(s.CONFIG['warmstart_checkpoint'])/'manifest.json').read_text())
        self.assertIn('resume.pt',manifest['files']);self.assertIn('bridge.safetensors',manifest['files'])

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    s.write(s.ROOT/'cpu-tests.json',{'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'pass':result.wasSuccessful(),'scope':'source/label/budget/checkpoint CPU contracts; not new quality scores'})
    raise SystemExit(not result.wasSuccessful())
