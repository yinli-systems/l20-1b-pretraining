"""Additional prose-content gate; does not censor political positions or news."""
import re
from continuation_quality import reject_reason as base_reject

VERSION = '20260911-general-v3'
EXPLICIT = re.compile(r'\b(?:porn(?:ography|ographic|hub)?|blowjobs?|handjobs?|cumshots?|'
                      r'gangbang|deepthroat|anal sex|sex with her|sex with him)\b',re.I)
SLANG_GLOSSARY = re.compile(r'\b\d+ definitions by\b',re.I)


def reject_reason(source, identifier, text, tokens):
    reason = base_reject(source,identifier,text,tokens)
    if reason or source == 'code':
        return reason
    if EXPLICIT.search(text) or SLANG_GLOSSARY.search(text):
        return 'explicit_sexual_or_slang_glossary'
    return None
