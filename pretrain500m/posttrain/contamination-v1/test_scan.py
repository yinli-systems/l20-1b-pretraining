import random
import string
import unicodedata

import pytest

from scan import Matcher, anchors, content_sha, normalize, POLICY


def reference(text, *, domain='multilingual_reading', field='question', rid='fixture'):
    return dict(reference_id=rid, training_admitted=False, domain=domain,
                provenance=dict(repo='owned-fixture/reference', path=rid+'.jsonl', row=0),
                fields=[dict(name=field, index=0, text=text, sha256=content_sha(text))])


DECOY = reference('An unrelated reference about volcanic islands and their geological formation across distant oceans.', rid='decoy')


@pytest.mark.parametrize('text', [
    'Café visitors discuss the historical architecture, public gardens and carefully preserved museum collections.',
    '小明在图书馆阅读关于自然科学历史文化城市建设环境保护教育发展艺术创作社会经济交通运输地理气候植物动物物理化学生物数学语言文字音乐绘画建筑设计的详细介绍，请问文章主要描述了哪些不同领域的知识以及它们之间的关系？',
    '図書館では自然科学と歴史文化に関する資料を整理しています。新しい本は教育研究や環境保護の取り組みについて詳しく説明し、地域社会の発展と日常生活との関係を考察しています。',
    'تحتوي المكتبة العامة على مجموعة متنوعة من الكتب والمخطوطات التي تتناول التاريخ والثقافة والعلوم والتعليم وحماية البيئة في مختلف المجتمعات.'
])
def test_unicode_and_whitespace_positive_controls(text):
    matcher = Matcher([reference(text)])
    variant = unicodedata.normalize('NFD', text).upper().replace(' ', '\n\t  ')
    hits, _ = matcher.matches('prefix ' + variant + ' suffix')
    assert hits
    assert hits[0]['reference_owners'][0]['reference_id'] == 'fixture'


def test_code_formatting_overlap_but_case_operator_and_number_are_distinct():
    code = 'def SelectPositive(records):\n    return [Record.value + 123 for Record in records if Record.value > 17]\n'
    assert 64 <= len(normalize(code, 'literal')) < 192
    matcher = Matcher([reference(code, domain='code', field='canonical_solution')])
    assert matcher.matches(code.replace('    ', '\t'))[0]
    for changed in [code.replace('SelectPositive', 'selectPositive'), code.replace('> 17', '< 17'),
                    code.replace('+ 123', '+ 124')]:
        assert not matcher.matches(changed)[0]


def test_math_compatibility_symbols_remain_distinct():
    text = 'Determine the complete expression for the transformation x² + 17y under the given constraints and boundary conditions.'
    matcher = Matcher([reference(text, domain='math', field='problem')])
    assert matcher.matches(text)[0]
    assert not matcher.matches(text.replace('²', '2'))[0]
    assert not matcher.matches(text.replace('17', '18'))[0]
    assert not matcher.matches(text.replace('x²', 'X²'))[0]


def test_short_primary_is_whole_document_only_and_answers_do_not_match():
    question = 'Compute the sum of 7 and 11.'
    matcher = Matcher([DECOY, reference(question), reference('18', field='answer', rid='answer')])
    assert matcher.matches(question)[0][0]['match_kind'] == 'whole_document'
    assert not matcher.matches('An unrelated lesson. ' + question + ' More discussion.')[0]
    assert not matcher.matches('18')[0]
    assert matcher.coverage['whole_document_only_fields'] == 1
    assert matcher.coverage['skipped_short_or_low_diversity_fields'] == 1


def test_fields_do_not_form_artificial_cross_field_matches():
    row = reference('One short passage concerns distant planets.')
    other = 'Another question discusses ancient manuscripts.'
    row['fields'].append(dict(name='question', index=1, text=other, sha256=content_sha(other)))
    matcher = Matcher([DECOY, row])
    assert not matcher.matches(row['fields'][0]['text']+' '+other)[0]


def test_long_anchor_coverage_and_documented_gap():
    rng = random.Random(20260914)
    text = ''.join(rng.choice(string.ascii_letters+string.digits) for _ in range(700))
    matcher = Matcher([reference(text, domain='code')])
    assert matcher.matches(text[17:17+287])[0]
    assert not matcher.matches(text[17:17+192])[0]
    assert text[-192:] in anchors(text)


def test_duplicate_patterns_retain_reference_owners_and_cap_saved_hits():
    text = 'A carefully prepared educational passage examines cultural traditions, scientific inquiry and historical evidence.'
    matcher = Matcher([reference(text, rid=str(i)) for i in range(8)])
    hits, _ = matcher.matches(text)
    assert hits[0]['reference_owner_count'] == 8
    assert len(hits[0]['reference_owners']) == POLICY['maximum_saved_owners_per_pattern']
    rng = random.Random(10)
    rows = [reference(''.join(rng.choice(string.ascii_letters) for _ in range(90)), domain='code', rid=str(i)) for i in range(40)]
    matcher = Matcher(rows)
    hits, truncated = matcher.matches(' '.join(r['fields'][0]['text'] for r in rows))
    assert truncated and len(hits) == POLICY['maximum_saved_hits_per_document']


def test_reference_identity_and_content_corruption_fail():
    row = reference('A sufficiently long reference passage establishes the relationship between language, knowledge and education.')
    with pytest.raises(ValueError, match='duplicate reference identity'):
        Matcher([row, row])
    row['fields'][0]['text'] += ' changed'
    with pytest.raises(ValueError, match='reference field digest mismatch'):
        Matcher([row])
