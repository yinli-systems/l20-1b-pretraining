import copy
import hashlib
import random
import struct
import unittest
from pathlib import Path
from features import PublicSuffix, canonical_url, repository, language_windows, minhash, features, POLICY


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.psl=PublicSuffix('com\norg\nco.uk\n*.ck\n!www.ck\nblogspot.com\n公司.cn\n')

    def test_suffix_wildcards_exceptions_private_and_idna(self):
        expected={'a.example.co.uk':'example.co.uk','a.b.ck':'a.b.ck','a.www.ck':'www.ck',
                  'x.author.blogspot.com':'author.blogspot.com','com':None,'www.食狮.公司.cn':'xn--85x722f.xn--55qx5d.cn'}
        for host,domain in expected.items():self.assertEqual(self.psl.domain(host),domain)

    def test_url_aliases_and_sensitive_path_separation(self):
        a=canonical_url('http://www.Example.com:80/%7Euser/a?utm_source=x&b=2&a=1#page=2',self.psl)
        b=canonical_url('https://example.com/~user/a?a=1&b=2',self.psl)
        self.assertEqual(a,b)
        self.assertNotEqual(canonical_url('https://example.com/a%2Fb',self.psl)[0],canonical_url('https://example.com/a/b',self.psl)[0])
        self.assertNotEqual(canonical_url('https://example.com/A',self.psl)[0],canonical_url('https://example.com/a',self.psl)[0])
        self.assertEqual(canonical_url('https://user:secret@example.com/a',self.psl),(None,None))

    def test_repository_case_and_git_suffix(self):
        self.assertEqual(repository('https://github.com/Owner/Repo.git'),'owner/repo')
        self.assertEqual(repository('OWNER/Repo'),'owner/repo')
        self.assertIsNone(repository('owner/repo/file.py'))

    def test_language_windows_include_tail(self):
        self.assertEqual(language_windows('short'),['short'])
        text='A'*3000+'B'*3000+'C'*3000
        windows=language_windows(text);self.assertEqual(len(windows),3)
        self.assertEqual(windows[-1],'C'*2048)

    def test_minhash_chunk_invariance_unicode_and_case_policy(self):
        text='The owned fixture discusses geometry, triangles, and equations. 数学推理需要完整步骤。'*7
        a,n=minhash(text,chunk_size=7);b,_=minhash(text,chunk_size=512)
        self.assertEqual(a,b);self.assertEqual(len(a),512);self.assertGreater(n,40)
        self.assertEqual(minhash(text.upper())[0],a)
        self.assertNotEqual(minhash('value = some_name + other_name; return value;',code=True)[0],
                            minhash('VALUE = SOME_NAME + OTHER_NAME; RETURN VALUE;',code=True)[0])

    def test_minhash_matches_scalar_unsigned_arithmetic(self):
        words='red blue green yellow orange purple black white'.split();prime=(1<<61)-1
        rng=random.Random(POLICY['seed']);a=[rng.randrange(1,prime) for _ in range(64)];b=[rng.randrange(0,prime) for _ in range(64)]
        hashes=[int.from_bytes(hashlib.blake2b('\x1f'.join(words[i:i+5]).encode(),digest_size=4).digest(),'little') for i in range(len(words)-4)]
        expected=[min((((h*aa+bb)&((1<<64)-1))%prime)&((1<<32)-1) for h in hashes) for aa,bb in zip(a,b)]
        self.assertEqual(minhash(' '.join(words))[0],struct.pack('<64I',*expected).hex())

    def test_preserves_math_symbols_and_numeric_template(self):
        row={'text':'A triangle has sides 3, 4 and 5. Compute x² + y² = z² and explain the result. '*4,
             'url':'https://example.com/problem','int_score':4}
        original=copy.deepcopy(row)
        f=features(row,'infiwebmath4',160,self.psl,lambda x:('en',0.99))
        self.assertEqual(row,original);self.assertNotIn('control_characters',f['flags'])
        changed=dict(row,text=row['text'].replace('3, 4 and 5','6, 8 and 10'))
        self.assertEqual(f['numeric_template_sha256'],features(changed,'infiwebmath4',160,self.psl,lambda x:('en',0.99))['numeric_template_sha256'])

    def test_code_license_and_syntax_gates(self):
        row={'text':'def square(value):\n    return value * value\n','repo_name':'Owned/Fixture',
             'license_type':'permissive','detected_licenses':['MIT'],'int_score':3}
        good=features(row,'code_python',80,self.psl,lambda x:('en',0.99))
        self.assertTrue(good['passes_declared_filters']);self.assertEqual(row['text'].splitlines()[1],'    return value * value')
        bad=dict(row,detected_licenses=['GPL-3.0'],text='def broken(:\n')
        result=features(bad,'code_python',80,self.psl,lambda x:('en',0.99))
        self.assertIn('code_license',result['flags']);self.assertIn('python_syntax',result['flags'])

    def test_wrong_language_garbled_and_repeated_prose_rejected(self):
        row={'text':('Repeated boilerplate text with enough characters for detection.\n'*20)+'\0',
             'url':'https://example.com/owned','language':'en','language_score':0.99}
        r=features(row,'dclm',300,self.psl,lambda x:('de',0.99))
        self.assertTrue({'repeated_long_lines','control_characters','language_windows'}<=set(r['flags']))

    def test_native_fasttext_adapter_when_remote_package_is_available(self):
        try:import fasttext
        except ImportError:self.skipTest('native wheel is qualified separately on the target CPython 3.12 runtime')
        from features import predict_language
        model=fasttext.load_model(str(Path(__file__).parent/'vendor/lid.176.bin'))
        texts={'en':'This is an English lesson about the history of science and how people learn new ideas.',
               'zh':'这是一篇中文教学文章，介绍数学学习的方法以及如何解释完整的推理过程。',
               'ja':'これは日本語の教育記事です。科学の歴史と新しい考え方を学ぶ方法について説明します。',
               'fr':'Ce texte français explique comment apprendre les mathématiques et comprendre les idées scientifiques.'}
        for language,text in texts.items():
            predicted,score=predict_language(model,text);self.assertEqual(predicted,language);self.assertGreater(score,0.60)


if __name__=='__main__':unittest.main()
