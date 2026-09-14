"""Declared quality features and family candidates; never training admission."""
import ast
from collections import Counter
import hashlib
import ipaddress
import math
import random
import re
import unicodedata
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

import numpy as np

TOKENS = re.compile(r'[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]|[^\W_]+|[^\w\s]', re.UNICODE)
NUMBERS = re.compile(r'\d+(?:[.,]\d+)*')
LICENSES = {'MIT','Apache-2.0','BSD-2-Clause','BSD-3-Clause','ISC','CC0-1.0','Unlicense'}
LANGUAGES = {'cmn_Hani':('cmn','zh'),'arb_Arab':('arb','ar'),'deu_Latn':('deu','de'),
             'fra_Latn':('fra','fr'),'jpn_Jpan':('jpn','ja'),'spa_Latn':('spa','es')}
POLICY = {'schema':'p529m-quality-family-features-v1','seed':20260914,
          'min_code_tokens':64,'min_other_tokens':128,'max_control_fraction':0.001,
          'max_replacement_fraction':0.001,'max_repeated_long_line_fraction':0.30,
          'repetition_min_line_characters':30,'max_dominant_character_fraction':0.40,
          'language_window_characters':2048,'language_window_positions':['start','middle','end'],
          'minimum_language_probability':0.60,'language_required_window_fraction':2/3,
          'minhash_permutations':64,'minhash_shingle_tokens':5,'minhash_chunk_shingles':512,
          'minhash_lsh_bands':16,'minhash_lsh_rows':4,
          'training_text_modified':False,'factual_correctness_verified':False,
          'candidate_generation_is_not_verified_near_duplicate_removal':True}


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


class PublicSuffix:
    def __init__(self,text):
        self.rules=set();self.exceptions=set();self.wildcards=set()
        for line in text.splitlines():
            rule=line.split('//',1)[0].strip()
            if not rule:continue
            dest=self.exceptions if rule.startswith('!') else self.wildcards if rule.startswith('*.') else self.rules
            rule=rule.removeprefix('!').removeprefix('*.').encode('idna').decode().lower()
            dest.add(rule)

    def domain(self,host):
        host=host.lower().rstrip('.').encode('idna').decode()
        try:return ipaddress.ip_address(host).compressed
        except ValueError:pass
        labels=host.split('.');suffix=1
        for i in range(len(labels)):
            tail='.'.join(labels[i:]);n=len(labels)-i
            if tail in self.exceptions:
                suffix=n-1;break
            if tail in self.rules:suffix=max(suffix,n)
            if i>0 and tail in self.wildcards:suffix=max(suffix,n+1)
        return '.'.join(labels[-(suffix+1):]) if len(labels)>suffix else None


def canonical_url(value,psl):
    try:
        p=urlsplit(value or '')
        if p.scheme not in ['http','https'] or not p.hostname or p.username or p.password:return None,None
        host=p.hostname.lower().rstrip('.').encode('idna').decode();port=p.port
        if host.startswith('www.'):host=host[4:]
        domain=psl.domain(host)
        authority='['+host+']' if ':' in host else host
        if port and (p.scheme,port) not in [('http',80),('https',443)]:authority+=':'+str(port)
        # Decode only unreserved ASCII characters, preserving encoded slashes and path case.
        path=re.sub(r'%([0-9A-Fa-f]{2})',lambda m:chr(int(m[1],16)) if chr(int(m[1],16)) in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~' else '%'+m[1].upper(),p.path or '/')
        query=sorted((k,v) for k,v in parse_qsl(p.query,keep_blank_values=True)
                     if not k.lower().startswith('utm_') and k.lower() not in ['fbclid','gclid'])
        return urlunsplit(('https',authority,path,urlencode(query),'')),domain
    except (ValueError,UnicodeError,AttributeError):return None,None


def repository(value):
    value=str(value or '').strip().strip('/')
    if value.startswith(('https://','http://')):
        p=urlsplit(value)
        if p.hostname not in ['github.com','www.github.com']:return None
        value=p.path.strip('/')
    value=value.removesuffix('.git').lower()
    return value if re.fullmatch(r'[a-z0-9_.-]+/[a-z0-9_.-]+',value) else None


def language_windows(text):
    size=POLICY['language_window_characters'];last=max(0,len(text)-size)
    return [text[i:i+size] for i in sorted({0,last//2,last})]


def predict_language(model,text):
    # The pinned fastText wrapper's np.array(copy=False) is incompatible with NumPy 2.
    # Use its same native prediction method and preserve the returned score/label.
    predictions=model.f.predict(' '.join(text.split())+'\n',1,0.0,'strict')
    return (predictions[0][1].removeprefix('__label__'),float(predictions[0][0])) if predictions else ('',0.0)


def minhash(text,code=False,chunk_size=None):
    text=unicodedata.normalize('NFC',text)
    if not code:text=text.casefold()
    tokens=TOKENS.findall(text)
    if len(tokens)<5:return None,len(tokens)
    rng=random.Random(POLICY['seed']);prime=np.uint64((1<<61)-1)
    a=np.array([rng.randrange(1,int(prime)) for _ in range(64)],dtype=np.uint64)
    b=np.array([rng.randrange(0,int(prime)) for _ in range(64)],dtype=np.uint64)
    result=np.full(64,(1<<32)-1,dtype=np.uint64);size=chunk_size or POLICY['minhash_chunk_shingles']
    for offset in range(0,len(tokens)-4,size):
        values=[int.from_bytes(hashlib.blake2b('\x1f'.join(tokens[i:i+5]).encode(),digest_size=4).digest(),'little')
                for i in range(offset,min(offset+size,len(tokens)-4))]
        h=np.asarray(values,dtype=np.uint64)[:,None]
        transformed=((h*a+b)%prime)&np.uint64((1<<32)-1)
        result=np.minimum(result,transformed.min(axis=0))
    return result.astype('<u4').tobytes().hex(),len(tokens)


def features(row,sid,encoded_tokens,psl,predict):
    text=row.get('text') or row.get('content');code=sid.startswith('code_');flags=[]
    if sid=='cosmopedia2':flags.append('excluded_from_f_series')
    if not isinstance(text,str) or not text.strip():raise ValueError('empty text')
    threshold=POLICY['min_code_tokens'] if code else POLICY['min_other_tokens']
    if encoded_tokens<threshold:flags.append('too_short')
    controls=sum(unicodedata.category(c)=='Cc' and c not in '\n\r\t\f' for c in text)
    replacements=text.count('\ufffd');length=len(text)
    if '\0' in text or controls/length>POLICY['max_control_fraction']:flags.append('control_characters')
    if replacements/length>POLICY['max_replacement_fraction']:flags.append('replacement_characters')
    long_lines=[re.sub(r'\s+',' ',l).strip() for l in text.splitlines() if len(l.strip())>=POLICY['repetition_min_line_characters']]
    lines=Counter(long_lines);repeat=sum(len(l)*(n-1) for l,n in lines.items())/length
    if repeat>POLICY['max_repeated_long_line_fraction']:flags.append('repeated_long_lines')
    chars=Counter(c for c in text if not c.isspace());dominant=max(chars.values(),default=0)/max(1,sum(chars.values()))
    if dominant>POLICY['max_dominant_character_fraction']:flags.append('dominant_character')
    url,domain=canonical_url(row.get('url'),psl);repo=repository(row.get('repo_name')) if code else None
    family='repository:'+repo if repo else 'url:'+url if url else None
    if family is None:flags.append('missing_family_identity')
    if not code and sid!='cosmopedia2' and domain is None:flags.append('missing_registrable_domain')
    if code:
        licenses=set(row.get('detected_licenses') or [])
        if row.get('license_type')!='permissive' or not licenses or not licenses<=LICENSES:flags.append('code_license')
        if int(row.get('int_score') or 0)<3:flags.append('code_education_score')
        if sid=='code_python':
            try:ast.parse(text)
            except (SyntaxError,ValueError):flags.append('python_syntax')
    if sid in ['finemath4','infiwebmath4'] and int(row.get('int_score') or 0)<4:flags.append('math_education_score')
    target=None;detected=[]
    if not code and sid!='cosmopedia2':
        target=LANGUAGES[sid.removeprefix('multilingual_')][1] if sid.startswith('multilingual_') else 'en'
        if sid.startswith('multilingual_'):
            expected=LANGUAGES[sid.removeprefix('multilingual_')][0]
            if row.get('language') not in [expected,sid.removeprefix('multilingual_')]:flags.append('source_language_label')
        elif sid=='dclm' and row.get('language')!='en':flags.append('source_language_label')
        elif sid=='pdf_en' and row.get('language') not in ['en','eng_Latn']:flags.append('source_language_label')
        score=row.get('language_score')
        if score is not None and (not math.isfinite(float(score)) or float(score)<(0.5 if sid=='finemath4' else 0.8)):
            flags.append('source_language_score')
        detected=[{'language':lang,'score':score} for lang,score in (predict(w) for w in language_windows(text))]
        acceptable=sum(v['language']==target and v['score']>=POLICY['minimum_language_probability'] for v in detected)
        if acceptable/len(detected)<POLICY['language_required_window_fraction']:flags.append('language_windows')
    template=None
    if sid in ['finemath4','infiwebmath4']:
        template=digest(NUMBERS.sub('<number>',re.sub(r'\s+',' ',unicodedata.normalize('NFC',text)).casefold()))
    sketch,shingle_tokens=minhash(text,code=code)
    return {'flags':sorted(set(flags)),'passes_declared_filters':not flags,'canonical_family':family,
        'canonical_url':url,'registrable_domain':domain,'repository':repo,'numeric_template_sha256':template,
        'controls':controls,'replacement_characters':replacements,'repeated_long_line_fraction':repeat,
        'dominant_character_fraction':dominant,'language_target':target,'language_windows':detected,
        'minhash64_u32_le_hex':sketch,'shingle_token_count':shingle_tokens,
        'factual_correctness_verified':False,'training_text_modified':False}
