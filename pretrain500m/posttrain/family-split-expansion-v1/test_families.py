import hashlib
import unittest
from families import DSU, candidates, join_keys, normalized, partition, select, shingles, verify


def row(i,source='dclm',family=None,passed=True):
    return dict(source_id=source,tranche=0,row=i,text_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
                normalized_text_sha256=None,legacy_normalized_family_sha256=None,canonical_family=family,
                numeric_template_sha256=None,passes_filters_and_bound_exclusions=passed,exclusion_reasons=[],
                minhash64_u32_le_hex=None,encoded_tokens_including_one_eos=1025)


class FamiliesTest(unittest.TestCase):
    def test_candidate_collision_does_not_certify_edge(self):
        records=[row(0),row(1)]
        for r in records:r['minhash64_u32_le_hex']='01'*256
        pairs,hot,_=candidates(records)
        self.assertEqual(pairs,[1]);self.assertFalse(hot)
        self.assertFalse(verify(shingles('one two three four five six'),shingles('six seven eight nine ten eleven'))[0])

    def test_near_threshold_and_unicode_preservation(self):
        a={('token',str(i)) for i in range(10)}
        self.assertTrue(verify(a,set(list(sorted(a))[:8]))[0])
        self.assertFalse(verify(a,set(list(sorted(a))[:7]))[0])
        self.assertTrue(shingles('中文数学数据集验证训练样本'))
        self.assertNotEqual(normalized('if a:\n  b()\nc()',True),normalized('if a:\n  b()\n  c()',True))
        self.assertNotEqual(normalized('VALUE',True),normalized('value',True))

    def test_hot_bucket_is_explicit_quarantine(self):
        records=[row(i) for i in range(4)]
        for r in records:r['minhash64_u32_le_hex']='01'*256
        pairs,hot,counts=candidates(records,bucket_limit=3)
        self.assertEqual(pairs,[]);self.assertEqual(hot,set(range(4)));self.assertEqual(counts,[1]*16)
        selected,_=select(records,DSU(4),DSU(4),hot)
        self.assertEqual(selected,set())

    def test_candidate_budget_fails_instead_of_truncation(self):
        records=[row(i) for i in range(4)]
        for r in records:r['minhash64_u32_le_hex']='01'*256
        with self.assertRaisesRegex(ValueError,'candidate budget'):candidates(records,pair_limit=2)

    def test_transitive_contamination_and_quality_failure_scope(self):
        records=[row(0,family='a'),row(1,family='a'),row(2,family='b'),row(3,family='c',passed=False),row(4,family='c')]
        records[0]['exclusion_reasons']=['verified_supplemental_exact_span_exclusion']
        families=DSU(5);duplicates=DSU(5);join_keys(records,families,duplicates);families.union(1,2)
        chosen,bad=select(records,families,duplicates,set())
        self.assertEqual(chosen,{4});self.assertIn(families.find(2),bad);self.assertNotIn(families.find(4),bad)

    def test_eligible_duplicate_owner_after_filters(self):
        records=[row(0,passed=False),row(1)];records[1]['text_sha256']=records[0]['text_sha256']
        families=DSU(2);duplicates=DSU(2);join_keys(records,families,duplicates)
        self.assertEqual(select(records,families,duplicates,set())[0],{1})

    def test_code_casefold_legacy_key_groups_but_does_not_dedup(self):
        records=[row(0,'code_python'),row(1,'code_python')]
        for r in records:r['legacy_normalized_family_sha256']='same'
        families=DSU(2);duplicates=DSU(2);join_keys(records,families,duplicates)
        self.assertEqual(families.find(0),families.find(1));self.assertNotEqual(duplicates.find(0),duplicates.find(1))

    def test_global_family_split_and_both_reserves(self):
        sources=['dclm','pdf_en','finemath4','code_python','multilingual_cmn_Hani']
        records=[row(i*10+j,s) for i,s in enumerate(sources) for j in range(6)]
        families=DSU(len(records));families.union(0,6)
        a,names,stats=partition(records,families,set(range(len(records))),minimum_families=2,minimum_tokens=1500)
        self.assertEqual(a[families.find(0)],a[families.find(6)])
        self.assertTrue(all(v==0 for p in stats['deficits'].values() for d in p.values() for v in d.values()))
        b,_,_=partition(records,families,set(reversed(range(len(records)))),minimum_families=2,minimum_tokens=1500)
        self.assertEqual(a,b)

    def test_insufficient_families_reported(self):
        records=[row(0)]
        _,_,stats=partition(records,DSU(1),{0},minimum_families=2,minimum_tokens=1500)
        self.assertGreater(sum(d['families'] for p in stats['deficits'].values() for d in p.values()),0)


if __name__=='__main__':unittest.main()
