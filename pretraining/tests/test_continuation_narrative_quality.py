import copy

import pytest

from continuation_narrative_quality import select_row,trim_boilerplate,reject_reason


def book():
    return {'id':'123','source':'project gutenberg','text':'A story with a beginning and an end.',
            'metadata':{'language':'en','license':'Public Domain','title':'Adventures in the Forest',
                        'url':'https://www.gutenberg.org/ebooks/123.txt.utf-8',
                        'provenance':'project_gutenberg-dolma-0000.json.gz:1'}}


def test_provenance_and_no_inplace_edit():
    row=book(); before=copy.deepcopy(row)
    selected,reason=select_row(row)
    assert reason is None and row==before
    assert selected['id']==row['metadata']['url']
    assert selected['metadata']['book_id']=='123'
    assert len(selected['metadata']['raw_text_sha256'])==64


@pytest.mark.parametrize('field,value',[('language','fr'),('license','Unknown'),
    ('url','https://example.com/ebooks/123.txt.utf-8'),('url','https://www.gutenberg.org/ebooks/124.txt.utf-8'),
    ('provenance','another-file:1'),('title','The Bible Story'),('title','Poems and Tales'),
    ('title','The Complete Works of Shakespeare'),('title','A Dictionary'),('title','The Federal Papers')])
def test_reject_wrong_metadata_or_forms(field,value):
    row=book(); row['metadata'][field]=value
    assert select_row(row)[0] is None


def test_only_exact_boilerplate_markers_are_trimmed():
    body='The story mentioned Project Gutenberg as part of a conversation.'
    marked='Header\n*** START OF THE PROJECT GUTENBERG EBOOK FOREST ***\n'+body+'\n*** END OF THE PROJECT GUTENBERG EBOOK FOREST ***\nLicense'
    assert trim_boilerplate(marked)==(body,None)
    assert trim_boilerplate(body)==(body,None)
    assert trim_boilerplate('He said "*** END OF THE PROJECT GUTENBERG EBOOK FOREST ***".')[1] is None
    assert trim_boilerplate('x'*9000+'\n*** START OF THE PROJECT GUTENBERG EBOOK FOREST ***\nBody')[0] is None


def test_short_and_noisy_prose_rejected():
    assert reject_reason('narrative','https://www.gutenberg.org/ebooks/123','short text',4)
    assert reject_reason('narrative','https://www.gutenberg.org/ebooks/123','x y z '*2000,6000)=='low_english_function_word_fraction'


def test_normal_prose_passes_structure_not_semantic_certification():
    # Unbroken prose avoids treating formatting alone as a semantic-quality test.
    text=' '.join(f'The traveller was in the forest and she walked with her friend through clearing number {i}.' for i in range(100))
    assert reject_reason('narrative','https://www.gutenberg.org/ebooks/123',text,2500) is None
