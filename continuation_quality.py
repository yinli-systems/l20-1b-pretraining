"""Conservative, auditable pilot filters; not a semantic correctness classifier.

The mathematical source allowlist deliberately trades recall for provenance.
It does not certify that every answer on an allowed site is correct.
"""
from collections import Counter
import re
from urllib.parse import urlparse

VERSION = '20260911-quality-v2'
MATH_HOSTS = {
    'math.stackexchange.com', 'physics.stackexchange.com', 'stats.stackexchange.com',
    'cs.stackexchange.com', 'cstheory.stackexchange.com', 'puzzling.stackexchange.com',
    'electronics.stackexchange.com', 'mathoverflow.net', 'physicsforums.com',
    'mathhelpforum.com', 'mathforum.org', 'mymathforum.com', 'tanyakhovanova.com',
    'openstax.org', 'libretexts.org', 'khanacademy.org', 'ck12.org', 'maths.org',
    'mathworld.wolfram.com', 'encyclopediaofmath.org', 'en.wikipedia.org',
    'proofwiki.org', 'themathdoctors.org', 'mathisfun.com', 'cut-the-knot.org',
    'betterexplained.com', 'brilliant.org', 'rosettacode.org',
}
MATH_SUFFIXES = ('.edu', '.ac.uk', '.edu.au', '.edu.in', '.edu.cn', '.ac.nz')
COMMERCIAL = re.compile(
    r'\b(?:sample term paper|ace my homework|(?:write|buy|order) my essay|'
    r'do my homework|buy essays?|hire (?:an? )?(?:essay|homework) writer|'
    r'plagiarism[- ]free (?:essay|paper)|essay writing service|'
    r'custom essay(?:s| writing)?|homework writing service)\b', re.I)
PRESS_RELEASE = re.compile(r'\b(?:PRNewswire|PR Newswire|PRWEB|Business Wire)\b', re.I)
WORD = re.compile(r'[A-Za-z]+')


def math_host_allowed(identifier):
    host = (urlparse(identifier).hostname or '').lower().rstrip('.')
    return bool(host) and (host.endswith(MATH_SUFFIXES) or any(
        host == allowed or host.endswith('.'+allowed) for allowed in MATH_HOSTS))


def reject_reason(source, identifier, text, tokens):
    """Return one ordered rejection reason, or None. Never rewrite the text."""
    if source == 'math' and not math_host_allowed(identifier):
        return 'math_provenance_not_allowlisted'
    if text.count('\ufffd') > max(2, len(text)*.003):
        return 'replacement_character_noise'
    if source == 'code':
        return None  # Preserve code syntax; do not apply prose heuristics to it.
    if tokens < 128 or len(WORD.findall(text)) < 70:
        return 'short_or_low_prose'
    if COMMERCIAL.search(text):
        return 'essay_homework_commercial'
    if source == 'dclm' and PRESS_RELEASE.search(text):
        return 'press_release_wire'
    if source in ('web', 'dclm'):
        units = [' '.join(x.lower().split()) for x in re.split(r'[\n.!?]+', text)]
        units = [x for x in units if len(x) >= 60]
        counts = Counter(units)
        repeated = sum(len(x)*(count-1) for x,count in counts.items() if count > 1)
        if repeated > .20*len(text):
            return 'repeated_prose'
    return None
