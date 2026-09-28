import hashlib
import numpy as np
import pytest

from continuation_quality import math_host_allowed, reject_reason
from curate_continuation import TokenStream, verified_documents, retain_sample


def test_allowlist_is_host_scoped():
    assert math_host_allowed('https://ocw.mit.edu/courses/math')
    assert math_host_allowed('https://math.stackexchange.com/questions/1')
    assert math_host_allowed('https://www.physicsforums.com/threads/1')
    for url in ('https://math.stackexchange.com.evil.test/q', 'not-a-url',
                'https://evil.test/mit.edu', 'https://domygmat.com/gmat-problem-solving-practice-2/',
                'https://pamsribbonsandroses.com/how-is-hmax-calculated/',
                'http://zhuravlova.me/post/diagramofseriescircuitforkids',
                'https://shinehandicraft.com/qa/question-whats-9-10-as-a-percentage.html'):
        assert not math_host_allowed(url)


def test_quality_reason_scope_and_structure():
    text = 'A coherent discussion of education and learning with ordinary prose. '*12
    assert reject_reason('math','https://math.stackexchange.com/q',text,200) is None
    assert reject_reason('dclm','id',text,200) == 'repeated_prose'
    assert reject_reason('web','id','Sample Term Paper '+text,200) == 'essay_homework_commercial'
    assert reject_reason('dclm','id','PRNewswire '+text,200) == 'press_release_wire'
    assert reject_reason('dclm','id','A short teaser.',92) == 'short_or_low_prose'
    assert reject_reason('code','id',text,200) is None


def test_document_reconstruction_cross_shard_and_literal_eos():
    # A literal EOS token inside a document is not mistaken for its boundary.
    docs = [[1,0,2],[3,4],[5,6,7,8]]
    def decode(ids):
        return ','.join(map(str,ids))
    rows = [(hashlib.sha256(decode(ids).encode()).digest(),str(i),len(ids)) for i,ids in enumerate(docs)]
    all_tokens = np.array(docs[0]+[0]+docs[1]+[0]+docs[2][:2],dtype=np.uint16)
    stream = TokenStream([all_tokens[:2],all_tokens[2:6],all_tokens[6:]])
    out = list(verified_documents(stream,rows,decode,lambda x:x))
    assert [x[1] for x in out] == ['0','1']
    assert stream.remaining == 0
    bad = [(b'incorrect', '0',3)]
    with pytest.raises(RuntimeError,match='SHA-256'):
        list(verified_documents(TokenStream([all_tokens[:4]]),bad,decode,lambda x:x))
    with pytest.raises(RuntimeError,match='after truncated'):
        list(verified_documents(TokenStream([all_tokens]),rows+[rows[0]],decode,lambda x:x))
    with pytest.raises(RuntimeError,match='no matching'):
        list(verified_documents(TokenStream([all_tokens]),rows[:2],decode,lambda x:x))
    # A repacked, block-aligned pool may drop more than one short trailing doc.
    extra_rows = rows+[rows[0]]
    out = list(verified_documents(TokenStream([all_tokens]),extra_rows,decode,lambda x:x,expected_tail=7))
    assert len(out) == 2
    with pytest.raises(RuntimeError,match='alignment tail'):
        list(verified_documents(TokenStream([all_tokens]),extra_rows,decode,lambda x:x,expected_tail=6))


def test_hash_priority_sample_is_order_independent():
    a, b = [], []
    values = [{'sha256':f'{i:064x}'} for i in range(100)]
    for value in values:
        retain_sample(a,value,4)
    for value in reversed(values):
        retain_sample(b,value,4)
    assert a == b == values[:4]


def test_web_provenance_and_aggregate_gate():
    from continuation_web_quality import host_allowed, reject_reason as web_reject
    assert host_allowed('https://www.nasa.gov/solar-system/example')
    assert host_allowed('https://openstax.org/books/physics/pages/1-1')
    assert not host_allowed('https://www.nasa.gov.evil.test/article')
    assert not host_allowed('<urn:uuid:missing-url>')
    assert web_reject('web','https://openstax.org/tag/math','content',200) == 'web_aggregate_url'


def test_general_prose_gate_preserves_nonsexual_discussion():
    from continuation_general_quality import reject_reason as general_reject
    prose = ' '.join(f'ordinary{i} discussion{i} learning{i}' for i in range(80))
    assert general_reject('dclm','id',prose,300) is None
    assert general_reject('dclm','id','3 definitions by someone '+prose,300) == 'explicit_sexual_or_slang_glossary'
    assert general_reject('dclm','id','political debate about equality '+prose,300) is None


def test_math_autosummary_gate():
    from continuation_math_quality import reject_reason as math_reject
    prose = ' '.join(f'ordinary{i} discussion{i} learning{i}' for i in range(80))
    assert math_reject('math','https://www.physicsforums.com/threads/x','In summary, '+prose,300) == 'forum_autosummary'
    assert math_reject('math','https://math.stackexchange.com/questions/x',prose,300) is None
