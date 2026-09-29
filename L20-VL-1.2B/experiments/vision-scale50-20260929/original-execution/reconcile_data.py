"""One-way repair of split metadata, not model training or benchmark relabelling.
Preserve both original holdout registries. Publish a new immutable derived corpus.
"""
from __future__ import annotations
import collections, hashlib, json, os, re, sqlite3, time
from decimal import Decimal
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'pilot-data'
SOURCE_SHA = 'fd5d4c600dcee971accbddff685a13d8f05c13c01213996827d76b0ccdd990f8'
SPLITS = frozenset(('train', 'development', 'confirmation', 'excluded'))
NUMERIC = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?')
# Fixed TRAIN-only review findings, declared before any new model experiment.
QUARANTINE = {
 '000524b1d175deef6669335f85fab9bc86a2e728c19e671d2271acb1f7b37427': 'subpixel/zero-height bars make the count visually unverifiable',
 '0147fe9b28215ff556f9c3916092fb60ff97f0507e76c8c2814898ad54f2ac03': 'download-promotion document rather than useful document QA',
}

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()

def reconcile(a: str, b: str) -> str:
    if a not in SPLITS or b not in SPLITS: raise ValueError('unknown split')
    if a == b: return a
    if 'excluded' in (a, b): return 'excluded'
    if a == 'train': return b
    if b == 'train': return a
    return 'excluded'  # Conflicting development/confirmation identities stay unused.

def numeric_label(text: str) -> str:
    candidate = text[:-1] if text.endswith('.') else text
    if text.endswith('.') and NUMERIC.fullmatch(candidate):
        value = Decimal(candidate.rstrip('%'))
        if not value.is_finite(): raise ValueError('nonfinite numeric label')
        return candidate
    return text

def write_json(path: Path, obj: object) -> None:
    data = (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    with path.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())

def verify_rows(rows: list[dict], original: list[dict], db: sqlite3.Connection) -> dict:
    by_id = {r['selection_sha256']: r for r in original}
    if len(rows) != len(by_id): raise ValueError('row identity loss')
    records = {h.hex(): (s, d.hex()) for h, s, d in db.execute('SELECT raw,split,document FROM images')}
    if len(records) != len(rows): raise ValueError('ledger row count')
    docs = {}
    for row in rows:
        old = by_id[row['selection_sha256']]
        if records[row['image_sha256']] != (row['split'], row['document_key']): raise ValueError('ledger mismatch')
        if docs.setdefault(row['document_key'], row['split']) != row['split']: raise ValueError('document leakage')
        if row['split'] == 'train' and (old['split'] != 'train' or row['previous_ledger_split'] != 'train'): raise ValueError('holdout to train')
        if old['split'] != 'train' and (row['response'] != old['response'] or row['references'] != old['references']): raise ValueError('heldout label changed')
        if row['image_path'] != old['image_path'] or row['image_sha256'] != old['image_sha256']: raise ValueError('image changed')
    return {'rows': len(rows), 'ledger_split_mismatches': 0, 'document_cross_split_conflicts': 0,
            'old_holdout_to_train': 0, 'original_heldout_labels_changed': 0}

def main() -> None:
    started = time.monotonic()
    if digest(SOURCE/'rows.json') != SOURCE_SHA: raise ValueError('source rows changed')
    out = ROOT/'prepared-v2'; tmp = ROOT/'prepared-v2.partial'
    if out.exists() or tmp.exists(): raise FileExistsError('no overwrite or implicit retry')
    old = json.loads((SOURCE/'rows.json').read_text())
    with sqlite3.connect('file:'+str(SOURCE/'identity.sqlite')+'?mode=ro',uri=True) as con:
        raw_records = list(con.execute('SELECT raw,pixel,document,split FROM images'))
    ledger = {raw.hex(): (pix.hex(),doc.hex(),split) for raw,pix,doc,split in raw_records}
    if len(ledger) != len(old): raise ValueError('old identities incomplete')
    document_splits = {}; source_conflicts = 0
    for row in old:
        pix,doc,old_split = ledger[row['image_sha256']]
        if (pix,doc) != (row['pixel224_sha256'],row['document_key']): raise ValueError('old identity mismatch')
        source_conflicts += row['split'] != old_split
        target = reconcile(row['split'],old_split)
        if row['selection_sha256'] in QUARANTINE:
            if row['split'] != 'train': raise ValueError('review excludes a holdout')
            target = 'excluded'
        document_splits[doc] = reconcile(document_splits.get(doc,target),target)
    fixed=[]; edits=[]; checked=set()
    for row in old:
        row=dict(row); previous=row['split']; ledger_split=ledger[row['image_sha256']][2]
        row.update(original_pilot_split=previous,previous_ledger_split=ledger_split,split=document_splits[row['document_key']])
        if row['selection_sha256'] in QUARANTINE: row['quarantine_reason']=QUARANTINE[row['selection_sha256']]
        if row['split']=='train':
            response=numeric_label(row['response'])
            if response!=row['response']:
                edits.append({'id':row['selection_sha256'],'before':row['response'],'after':response,'reason':'entire numeric label trailing punctuation only'})
                row['response']=response;row['references']=[response];row['format_normalized']=True
        path=Path(row['image_path'])
        if path.is_symlink() or not path.resolve().is_relative_to((SOURCE/'images').resolve()): raise ValueError('image outside original corpus')
        if row['image_sha256'] not in checked:
            if digest(path)!=row['image_sha256']: raise ValueError('image hash mismatch')
            checked.add(row['image_sha256'])
        fixed.append(row)
    tmp.mkdir()
    db=sqlite3.connect(tmp/'identity.sqlite');db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE images(seq INTEGER PRIMARY KEY, raw BLOB UNIQUE NOT NULL, pixel BLOB UNIQUE NOT NULL, document BLOB NOT NULL, split TEXT NOT NULL, source TEXT NOT NULL)')
    db.execute('CREATE INDEX document_idx ON images(document)')
    db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    policy={'policy':'union-of-prior-holdouts-v2','source_rows_sha256':SOURCE_SHA,'source_split_mismatches':source_conflicts,
            'holdout_to_train_allowed':False,'dev_confirmation_conflict':'excluded','manual_train_quarantine':QUARANTINE}
    for r in fixed:
        db.execute('INSERT INTO images(raw,pixel,document,split,source) VALUES(?,?,?,?,?)',(bytes.fromhex(r['image_sha256']),bytes.fromhex(r['pixel224_sha256']),bytes.fromhex(r['document_key']),r['split'],r['source']))
    db.execute('INSERT INTO metadata VALUES(?,?)',('frozen_split_policy',json.dumps(policy,sort_keys=True)));db.commit()
    checks=verify_rows(fixed,old,db);db.close()
    write_json(tmp/'rows.json',fixed);write_json(tmp/'format-edits.json',edits);write_json(tmp/'split-policy.json',policy)
    report={'status':'data_integrity_verified_not_training_release','checks':checks,'source_conflicts_fixed':source_conflicts,
       'counts':dict(collections.Counter(r['domain']+'/'+r['split'] for r in fixed)), 'raw_images_hash_verified':len(checked),
       'training_numeric_labels_canonicalized':len(edits),'full_semantic_label_audit':False,
       'original_source_rows_unchanged':digest(SOURCE/'rows.json')==SOURCE_SHA,'bulk_training_authorized':False,
       'training_updates':0,'parent_and_all_checkpoints_unchanged':True,
       'files':{n:digest(tmp/n) for n in ['rows.json','identity.sqlite','format-edits.json','split-policy.json']},
       'source_program_sha256':digest(Path(__file__)),'seconds':time.monotonic()-started,'created_unix':time.time()}
    write_json(tmp/'admission.json',report);os.replace(tmp,out)
    fd=os.open(str(ROOT),os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()
