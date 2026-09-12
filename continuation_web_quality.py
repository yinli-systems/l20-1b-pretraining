"""Pilot-specific provenance restriction after semantic audit rejected v2 web.

This is not an assertion that every allowed page is factual, or that excluded
publishers are bad. Broad conversational coverage remains in the DCLM component.
"""
import re
from urllib.parse import urlparse
from continuation_quality import reject_reason as base_reject

VERSION = '20260911-web-provenance-v3'
HOSTS = {
    'en.wikipedia.org', 'en.wikibooks.org', 'en.wikisource.org', 'openstax.org',
    'libretexts.org', 'khanacademy.org', 'ck12.org', 'britannica.com',
    'bbc.co.uk', 'bbc.com', 'smithsonianmag.com', 'nationalgeographic.com',
    'scientificamerican.com', 'sciencenews.org', 'sciencedaily.com',
    'quantamagazine.org', 'worldhistory.org', 'science.org', 'nature.com',
    'rsc.org', 'aps.org', 'ams.org', 'royalsociety.org', 'nhs.uk',
    'oercommons.org', 'europeana.eu', 'si.edu', 'nctm.org', 'readwritethink.org',
}
SUFFIXES = ('.edu', '.ac.uk', '.edu.au', '.edu.in', '.edu.cn', '.ac.nz',
            '.gov', '.gov.uk', '.gov.au', '.gc.ca', '.sch.uk')
AGGREGATE_PATH = re.compile(r'/(?:tag|tags|search|category|categories|feed|rss)(?:/|\?|$)',re.I)


def host_allowed(identifier):
    host = (urlparse(identifier).hostname or '').lower().rstrip('.')
    return bool(host) and (host.endswith(SUFFIXES) or any(
        host == item or host.endswith('.'+item) for item in HOSTS))


def reject_reason(source, identifier, text, tokens):
    if not host_allowed(identifier):
        return 'web_provenance_not_allowlisted'
    if AGGREGATE_PATH.search(urlparse(identifier).path):
        return 'web_aggregate_url'
    return base_reject('web',identifier,text,tokens)
