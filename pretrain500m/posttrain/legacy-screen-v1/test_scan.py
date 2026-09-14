import ast
import hashlib
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import re
import tempfile
import unittest
import unicodedata

from scan import first_legacy_match, legacy_normalized_hash
import scan


class LegacyCompatibility(unittest.TestCase):
    def setUp(self):
        # Execute only the two named pure functions from the actual original packer.
        path = Path(__file__).parent / 'original_pack_data.py'
        if not path.exists():
            path = Path(__file__).resolve().parents[2] / 'pack_data.py'
        tree = ast.parse(path.read_text())
        names = {'normalized_text', 'is_contaminated'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual(len(nodes), 2)
        env = dict(re=re, hashlib=hashlib, unicodedata=unicodedata, WORD_RE=re.compile(r'[a-z0-9]+'))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
        self.original = env

    def test_legacy_matching_parity_and_offsets(self):
        phrase = 'one two three four five six seven eight nine ten eleven twelve thirteen'
        h = hashlib.blake2b(phrase.encode(), digest_size=8).digest()
        for text in [phrase, phrase.upper(), 'prefix '+phrase+' suffix', 'one two', '',
                     '中文 数学 '+phrase.replace(' ', '\n'), phrase.replace('one', 'different')]:
            with self.subTest(text=text):
                self.assertEqual(first_legacy_match(text, {h}) is not None,
                                 self.original['is_contaminated'](text, {h}))
        self.assertEqual(first_legacy_match('prefix '+phrase, {h})['word_offset'], 1)

    def test_old_normalization_parity_without_casefold(self):
        for text in ['  ＡＢＣ\nDef ', 'Straße', 'STRASSE', 'Ａ数学\t\tＢ', 'İstanbul', 'e\u0301']:
            expected = hashlib.sha256(self.original['normalized_text'](text).encode()).hexdigest()
            self.assertEqual(legacy_normalized_hash(text), expected)
        self.assertNotEqual(legacy_normalized_hash('Straße'), legacy_normalized_hash('STRASSE'))

    def test_thirteen_word_boundary_and_empty_reference(self):
        text = ' '.join(str(x) for x in range(13))
        h = hashlib.blake2b(text.encode(), digest_size=8).digest()
        self.assertIsNotNone(first_legacy_match(text, {h}))
        self.assertIsNone(first_legacy_match(' '.join(str(x) for x in range(12)), {h}))
        self.assertIsNone(first_legacy_match(text, set()))

    def test_parallel_file_scan_preserves_each_row_and_candidate(self):
        phrase = ' '.join('fixture'+str(i) for i in range(13))
        scan.HASHES = frozenset({hashlib.blake2b(phrase.encode(), digest_size=8).digest()})
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); work = []
            for i in range(2):
                raw=root/f'{i}.jsonl.gz'
                with gzip.open(raw, 'wt') as f:
                    for text in ['A clean owned document', phrase]:
                        f.write(json.dumps({'text':text,'_provenance':{'owned_fixture':i}})+'\n')
                work.append(('fixture',i,str(raw),2,str(root)))
            with mp.get_context('fork').Pool(2) as pool:
                results=pool.map(scan.scan_file,work)
            self.assertEqual(sum(r['rows'] for r in results),4)
            self.assertEqual(sum(r['candidate_rows'] for r in results),2)
            for r in results:
                with gzip.open(r['index'],'rt') as f:rows=[json.loads(l) for l in f]
                self.assertEqual([v['row'] for v in rows],[0,1])
                self.assertEqual(rows[1]['text_sha256'],hashlib.sha256(phrase.encode()).hexdigest())

    def test_changed_row_count_fails_before_completion(self):
        scan.HASHES=frozenset()
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);raw=root/'raw.jsonl.gz'
            with gzip.open(raw,'wt') as f:f.write(json.dumps({'text':'owned fixture'})+'\n')
            with self.assertRaisesRegex(ValueError,'row count changed'):
                scan.scan_file(('fixture',0,str(raw),2,str(root)))
            self.assertFalse((root/'0-fixture.progress.json').exists())


if __name__ == '__main__':
    unittest.main()
