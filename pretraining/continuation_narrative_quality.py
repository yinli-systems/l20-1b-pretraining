"""Conservative title/structure admission for one pinned Gutenberg source.

Title cues are NOT authoritative genre labels. This gate trades coverage for
auditability; accepted samples still require semantic review before training.
Upstream Public Domain metadata is retained, not independently legally certified.
"""
from collections import Counter
import hashlib
import re
from urllib.parse import urlparse

from continuation_quality import reject_reason as base_reject

VERSION = '20260911-narrative-title-prose-v1'
POSITIVE = re.compile(r'\b(?:novels?|novellas?|stor(?:y|ies)|tales?|adventures?|myster(?:y|ies)|'
                      r'romances?|memoirs?|recollections?|autobiograph(?:y|ies))\b',re.I)
NEGATIVE = re.compile(r'\b(?:complete works|bible|scriptures?|testament|psalms?|poems?|poetry|verse|'
                      r'dramas?|plays?|sermons?|prayers?|hymns?|magazines?|journals?|periodicals?|'
                      r'dictionar(?:y|ies)|encyclop(?:ae|e)dia|catalog(?:ue)?s?|cookery|cookbooks?|'
                      r'recipes?|constitution|declaration|statutes?)\b',re.I)
START = re.compile(r'^\s*\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK [^\r\n]+?\*\*\*[ \t]*\r?$',re.M|re.I)
END = re.compile(r'^\s*\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK [^\r\n]+?\*\*\*[ \t]*\r?$',re.M|re.I)
WORDS = re.compile(r'[A-Za-z]+')
COMMON = set('the and of to in a was he she it that his her with as for had on at by not from they you I is'.lower().split())


def trim_boilerplate(text):
    starts,ends = list(START.finditer(text)),list(END.finditer(text))
    if len(starts)>1 or len(ends)>1:
        return None,'ambiguous_gutenberg_markers'
    if starts and starts[0].start()>8000:
        return None,'interior_gutenberg_start_marker'
    if ends and ends[0].start()<.8*len(text) and len(text)-ends[0].start()>25000:
        return None,'interior_gutenberg_end_marker'
    begin = starts[0].end() if starts else 0
    finish = ends[0].start() if ends else len(text)
    if finish<=begin:
        return None,'invalid_gutenberg_marker_order'
    return text[begin:finish].strip(),None


def select_row(row):
    if not isinstance(row,dict) or not isinstance(row.get('metadata'),dict):
        return None,'missing_metadata'
    meta = row['metadata']
    if row.get('source','').strip().lower()!='project gutenberg':
        return None,'wrong_source'
    if meta.get('language')!='en':
        return None,'not_english_metadata'
    if str(meta.get('license','')).strip().lower()!='public domain':
        return None,'license_metadata_not_expected'
    url = str(meta.get('url',''))
    parsed = urlparse(url)
    path = re.fullmatch(r'/ebooks/(\d+)(?:\.txt\.utf-8)?',parsed.path)
    if parsed.scheme!='https' or parsed.hostname not in ('www.gutenberg.org','gutenberg.org') or not path or path[1]!=str(row.get('id')):
        return None,'provenance_url_mismatch'
    if not str(meta.get('provenance','')).startswith('project_gutenberg-dolma-0000.json.gz:'):
        return None,'wrong_source_shard'
    title = str(meta.get('title',''))
    if NEGATIVE.search(title):
        return None,'excluded_title_form'
    if not POSITIVE.search(title):
        return None,'no_positive_narrative_title_cue'
    raw = row.get('text')
    if not isinstance(raw,str) or not raw.strip():
        return None,'missing_text'
    text,reason = trim_boilerplate(raw)
    if reason:
        return None,reason
    return {'id':url,'source':'narrative','text':text,
            'metadata':{**meta,'book_id':str(row['id']),
                        'raw_text_sha256':hashlib.sha256(raw.encode()).hexdigest(),
                        'cleaned_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                        'exact_marker_trimmed':text!=raw.strip(),
                        'selection':'title cues plus prose structure, not an authoritative genre annotation'}},None


def reject_reason(source,identifier,text,tokens):
    reason = base_reject(source,identifier,text,tokens)
    if reason:
        return reason
    if source!='narrative':
        return 'wrong_narrative_source'
    words = WORDS.findall(text)
    if tokens<2048 or len(words)<1000:
        return 'short_narrative'
    if sum(word.lower() in COMMON for word in words)<.07*len(words):
        return 'low_english_function_word_fraction'
    if sum(len(w) for w in words)<.92*sum(c.isalpha() for c in text):
        return 'non_latin_prose_fraction'
    if sum(len(w)==1 and w.lower() not in ('a','i') for w in words)>.12*len(words):
        return 'fragmented_ocr_words'
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines)>100 and sum(len(x)<35 for x in lines)>.65*len(lines):
        return 'predominantly_short_lines'
    paragraphs = [' '.join(x.lower().split()) for x in re.split(r'\n\s*\n',text)]
    repeated = sum(len(x)*(n-1) for x,n in Counter(x for x in paragraphs if len(x)>100).items() if n>1)
    if repeated>.15*len(text):
        return 'repeated_paragraphs'
    return None
