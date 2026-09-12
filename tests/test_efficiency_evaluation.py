"""Fail-closed inventory, artifact, disk and paired-protocol checks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from discover_efficiency_baselines import inference_files
import run_efficiency_evaluation as runner
from run_efficiency_evaluation import (PROTOCOL, read_plan, verify_file, require_space,
                                       release_owned_downloads, verify_converted_weights,
                                       append_download_response)
from summarize_efficiency_frontier import apparent_frontier, render_svg


def artifact(name='config.json', data=b'{}'):
    return {'path':name,'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),
            'git_blob_sha1':hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()}


class EfficiencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def plan(self):
        return {'protocol':copy.deepcopy(PROTOCOL),'program_sha256':{
            name:runner.sha256(Path(runner.__file__).with_name(name)) for name in runner.REQUIRED_PROGRAMS},'jobs':[
            {'id':'model_1','repo':'author/model','revision':'a'*40,'files':[artifact()],
             'adapter':'transformers_builtin','training_tokens':20_000_000_000}]}

    def read(self, data):
        path = self.root/'plan.json'
        path.write_text(json.dumps(data))
        return read_plan(path)

    def test_valid_plan(self):
        self.assertEqual(len(self.read(self.plan())['jobs']),1)

    def test_inexact_checkpoint_loading_rejected(self):
        runner.verify_loading_info({'missing_keys':[],'unexpected_keys':[],'mismatched_keys':[],'error_msgs':[]})
        for field in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.verify_loading_info({field:['bad']})

    def test_abnormal_B_stop_never_starts_model_download(self):
        root=self.root/'continuation';root.mkdir()
        (root/'gpu.lock').touch()
        pilot=root/'pilot-B';pilot.mkdir()
        (pilot/'status.json').write_text(json.dumps({'stage':'stopped_preservation_gate','branch_step':50}))
        (pilot/'resume.json').write_text(json.dumps({'step':50,'sha256':'x'}))
        plan_path=self.root/'plan.json';plan_path.write_text(json.dumps(self.plan()))
        out=self.root/'evaluations'
        with patch.object(runner,'OUT',out), patch.object(runner,'CONTINUATION',root), \
             patch.object(runner,'validate_runtime'), patch.object(runner,'validate_adapter'), \
             patch.object(runner,'download_model') as download, patch.object(runner.subprocess,'check_output') as gpu:
            with self.assertRaisesRegex(RuntimeError,'B did not finish normally'):
                runner.queue(self.plan(),plan_path)
            download.assert_not_called();gpu.assert_not_called()

    def test_explicit_checkpointed_pause_admitted_only_with_exact_receipt(self):
        checkpoint={'step':61,'sha256':'abc'}
        p={'authorized_training_pause':{'reason':'user_requested_evaluation_priority','step':61,'checkpoint':checkpoint}}
        with patch.object(runner,'sha256',return_value='abc'):
            runner.admit_B_checkpoint(p,{'stage':'checkpointed_stop','branch_step':61},checkpoint)
            with self.assertRaises(RuntimeError):
                runner.admit_B_checkpoint(p,{'stage':'stopped_preservation_gate','branch_step':61},checkpoint)
            with self.assertRaises(RuntimeError):
                runner.admit_B_checkpoint(p,{'stage':'checkpointed_stop','branch_step':62},checkpoint)
            with self.assertRaises(RuntimeError):
                runner.admit_B_checkpoint({}, {'stage':'checkpointed_stop','branch_step':61},checkpoint)

    def test_mutable_revision_rejected(self):
        p=self.plan();p['jobs'][0]['revision']='main'
        with self.assertRaises(ValueError):self.read(p)

    def test_incomplete_program_bindings_rejected(self):
        p=self.plan();p['program_sha256']={}
        with self.assertRaises(ValueError):self.read(p)

    def test_runtime_conversion_must_be_source_and_program_bound(self):
        p=self.plan();source=artifact('pytorch_model.bin',b'source');p['jobs'][0]['files'].append(source)
        p['runtime_weight_conversions']={'model_1':{'source_path':'pytorch_model.bin',
            'source_sha256':source['sha256'],'output_path':'model.safetensors',
            'expected_tensors':201,'expected_numel':1_100_048_384,
            'container_image':'python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea'}}
        with self.assertRaises(ValueError):self.read(p)
        p['program_sha256'][runner.CONVERTER_PROGRAM]=runner.sha256(Path(runner.__file__).with_name(runner.CONVERTER_PROGRAM))
        self.assertEqual(self.read(p)['runtime_weight_conversions']['model_1']['source_sha256'],source['sha256'])
        p['runtime_weight_conversions']['model_1']['source_sha256']='0'*64
        with self.assertRaises(ValueError):self.read(p)

    def test_converted_weight_receipt_and_content_are_verified(self):
        directory=self.root/'model';directory.mkdir()
        source=artifact('pytorch_model.bin',b'source');(directory/'pytorch_model.bin').write_bytes(b'source')
        output=b'converted';(directory/'model.safetensors').write_bytes(output)
        receipt={'status':'exact_tensor_equality_verified','source_sha256':source['sha256'],
                 'tensor_count':201,'numel':1_100_048_384,'dtype':'float32',
                 'output_bytes':len(output),'output_sha256':hashlib.sha256(output).hexdigest(),
                 'required_launcher_boundary':'network=none, read-only root, cap-drop=all, no-new-privileges, pids<=64'}
        (directory/'conversion-receipt.json').write_text(json.dumps(receipt))
        job={'id':'model_1','files':[source]};plan={'runtime_weight_conversions':{'model_1':{
            'source_path':'pytorch_model.bin','source_sha256':source['sha256'],'output_path':'model.safetensors',
            'expected_tensors':201,'expected_numel':1_100_048_384}}}
        self.assertEqual(verify_converted_weights(plan,job,directory)['sha256'],receipt['output_sha256'])
        (directory/'model.safetensors').write_bytes(b'changed')
        with self.assertRaises(ValueError):verify_converted_weights(plan,job,directory)

    def test_author_adapter_requires_bound_files(self):
        p=self.plan();p['jobs'][0]['adapter']='author_hf_olmo'
        with self.assertRaises(ValueError):self.read(p)

    def test_unknown_training_compute_rejected(self):
        p=self.plan();p['jobs'][0]['training_tokens']=None
        with self.assertRaises(ValueError):self.read(p)

    def test_unsafe_names_rejected(self):
        for name in ['../weights.bin','/config.json','.', '..']:
            p=self.plan();p['jobs'][0]['files'][0]['path']=name
            with self.subTest(name=name), self.assertRaises(ValueError):self.read(p)

    def test_duplicate_jobs_rejected(self):
        p=self.plan();p['jobs']*=2
        with self.assertRaises(ValueError):self.read(p)

    def test_duplicate_files_rejected(self):
        p=self.plan();p['jobs'][0]['files']*=2
        with self.assertRaises(ValueError):self.read(p)

    def test_shots_precision_and_limit_frozen(self):
        for key,value in [('num_fewshot',5),('dtype','float32'),('limit',100),('apply_chat_template',True)]:
            p=self.plan();p['protocol'][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):self.read(p)

    def test_lfs_and_git_artifact_verification(self):
        path=self.root/'config.json';path.write_bytes(b'{}')
        item=artifact();verify_file(path,item)
        item['sha256']=None;verify_file(path,item)

    def test_artifact_mutation_and_symlink_rejected(self):
        path=self.root/'config.json';path.write_bytes(b'[]')
        with self.assertRaises(ValueError):verify_file(path,artifact())
        link=self.root/'link';link.symlink_to(path)
        with self.assertRaises(ValueError):verify_file(link,artifact(data=b'[]'))

    def test_disk_reserve_not_just_download_size(self):
        from collections import namedtuple
        usage=namedtuple('Usage','total used free')(1000,500,500)
        with patch('run_efficiency_evaluation.shutil.disk_usage',return_value=usage):
            self.assertEqual(require_space(self.root,100,400),500)
            with self.assertRaises(RuntimeError):require_space(self.root,101,400)

    def test_resumed_download_requires_exact_content_range_and_appends(self):
        class Response:
            status_code = 206
            headers = {'Content-Range':'bytes 3-5/6','Content-Length':'3'}
            def raise_for_status(self): pass
            def iter_content(self, _): return iter((b'de',b'f'))
        path = self.root/'artifact.part';path.write_bytes(b'abc')
        self.assertEqual(append_download_response(Response(),path,6,3),6)
        self.assertEqual(path.read_bytes(),b'abcdef')

    def test_resumed_download_rejects_wrong_range_without_writing(self):
        class Response:
            status_code = 206
            headers = {'Content-Range':'bytes 2-5/6','Content-Length':'4'}
            def raise_for_status(self): pass
            def iter_content(self, _): return iter((b'cdef',))
        path = self.root/'artifact.part';path.write_bytes(b'abc')
        with self.assertRaisesRegex(ValueError,'Unexpected resumed byte range'):
            append_download_response(Response(),path,6,3)
        self.assertEqual(path.read_bytes(),b'abc')

    def test_full_response_restarts_partial_only_with_exact_length(self):
        class Response:
            status_code = 200
            headers = {'Content-Length':'6'}
            def raise_for_status(self): pass
            def iter_content(self, _): return iter((b'abcdef',))
        path = self.root/'artifact.part';path.write_bytes(b'abc')
        self.assertEqual(append_download_response(Response(),path,6,3),6)
        self.assertEqual(path.read_bytes(),b'abcdef')

    def test_weight_format_avoids_duplicate_and_training_artifacts(self):
        names=['model.safetensors','pytorch_model.bin','optimizer.pt','config.json','evals/results.json']
        siblings=[{'rfilename':name,'size':5,'blobId':'a'*40} for name in names]
        files=inference_files(siblings)
        self.assertEqual({x['path'] for x in files},{'model.safetensors','config.json'})

    def test_missing_weights_not_admitted(self):
        with self.assertRaises(ValueError):inference_files([{'rfilename':'config.json','size':1,'blobId':'a'*40}])

    def test_cleanup_only_owned_verified_files(self):
        directory=self.root/'models'/'model_1';directory.mkdir(parents=True)
        item=artifact();(directory/'config.json').write_bytes(b'{}')
        job={'id':'model_1','repo':'author/model','revision':'a'*40,'files':[item]}
        (directory/'ownership.json').write_text(json.dumps({'job':job['id'],'repo':job['repo'],'revision':job['revision'],'files':job['files']}))
        keep=directory/'unrelated.txt';keep.write_text('retain')
        with patch('run_efficiency_evaluation.OUT',self.root):
            removed=release_owned_downloads(job,directory)
        self.assertEqual(removed,[str(directory/'config.json')]);self.assertTrue(keep.exists())

    def test_cleanup_checks_all_hashes_before_any_delete(self):
        directory=self.root/'models'/'model_1';directory.mkdir(parents=True)
        good=artifact();bad=artifact('other.json')
        (directory/'config.json').write_bytes(b'{}');(directory/'other.json').write_bytes(b'[]')
        job={'id':'model_1','repo':'author/model','revision':'a'*40,'files':[good,bad]}
        (directory/'ownership.json').write_text(json.dumps({'job':job['id'],'repo':job['repo'],'revision':job['revision'],'files':job['files']}))
        with patch('run_efficiency_evaluation.OUT',self.root), self.assertRaises(ValueError):
            release_owned_downloads(job,directory)
        self.assertTrue((directory/'config.json').exists())

    def test_frontier_requires_no_more_compute_and_no_worse_score(self):
        points=[{'id':'a','compute':1e20,'score':50,'category':'natural_corpus_base'},
                {'id':'b','compute':2e20,'score':49,'category':'natural_corpus_base'},
                {'id':'c','compute':3e20,'score':55,'category':'natural_corpus_base'},
                {'id':'teacher','compute':1e19,'score':80,'category':'teacher_synthetic_reference'}]
        self.assertEqual(apparent_frontier(points),['a','c'])
        svg=render_svg(points)
        self.assertIn('not global SOTA',svg)
        self.assertIn('teacher',svg)

    def test_svg_escapes_model_labels(self):
        points=[{'id':'<bad&>','compute':1e20,'score':50,'category':'natural_corpus_base'}]
        svg=render_svg(points)
        self.assertIn('&lt;bad&amp;&gt;',svg)
        self.assertNotIn('<bad&>',svg)


if __name__=='__main__':unittest.main()
