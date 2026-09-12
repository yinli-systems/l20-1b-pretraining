"""Refine both intact math pool segments, preserving their independent boundaries."""
import json
import sqlite3

from continuation_common import RUN, BLOCK, GLOBAL, MIXES, PILOT_STEPS, atomic_json, lock, quotas, sha256
from continuation_math_quality import reject_reason
from curate_continuation import curate


def main():
    handle = lock(RUN/'refine-math.lock')
    original = RUN/'curated-v2/math'
    first = RUN/'refined-v3/math-original'
    curate('math',input_path=original/'manifest.json',directory=first,
           input_inventory=original/'documents.sqlite',quality_filter=reject_reason)
    m = json.loads((RUN/'refill-v2/math/manifest.json').read_text())
    offset = json.loads((RUN/'data/math/manifest.json').read_text())['counters']['accepted']
    count = m['counters']['accepted']
    wrapper = RUN/'refined-v3/math-refill-input'
    wrapper.mkdir(parents=True,exist_ok=True)
    inventory = wrapper/'documents.sqlite'
    if not inventory.exists():
        old = sqlite3.connect(f'file:{RUN}/fresh-dedup.sqlite?mode=ro',uri=True)
        if old.execute("SELECT COUNT(*) FROM documents WHERE source='math'").fetchone()[0] != offset+count:
            raise RuntimeError('math source inventory changed; cannot assume segment bounds')
        new = sqlite3.connect(inventory)
        new.execute('CREATE TABLE documents(hash BLOB PRIMARY KEY,id TEXT,tokens INTEGER,reason TEXT)')
        rows = old.execute("SELECT hash,id,tokens,NULL FROM documents WHERE source='math' ORDER BY rowid LIMIT ? OFFSET ?",(count,offset))
        new.executemany('INSERT INTO documents VALUES(?,?,?,?)',rows)
        new.commit()
        new.close()
        old.close()
    receipt = {**m,'inventory_sha256':sha256(inventory),
               'original_refill_manifest_sha256':sha256(RUN/'refill-v2/math/manifest.json')}
    atomic_json(wrapper/'manifest.json',receipt)
    second = RUN/'refined-v3/math-refill'
    curate('math',input_path=wrapper/'manifest.json',directory=second,
           input_inventory=inventory,quality_filter=reject_reason)
    parts = [first/'manifest.json',second/'manifest.json']
    manifests = [json.loads(p.read_text()) for p in parts]
    output = {k:manifests[0][k] for k in ('source','base_inputs_sha256','benchmark_sha256','tokenizer_sha256',
                                       'source_revisions','quality_version','quality_policy_sha256')}
    output.update(shards=[r for m in manifests for r in m['shards']],tokens=sum(m['tokens'] for m in manifests),
                  input_manifests=[{'path':str(p),'sha256':sha256(p)} for p in parts],stage='curated_pending_review')
    required = quotas(MIXES['A'],GLOBAL*PILOT_STEPS)['fresh_math']*BLOCK
    if output['tokens'] < required:
        raise RuntimeError('math refinement leaves too few blocks even for pilot A')
    atomic_json(RUN/'refined-v3/math/manifest.json',output)
    print(json.dumps({'stage':'refined_math_ready','tokens':output['tokens'],'pilot_A_required':required}),flush=True)
    handle.close()


if __name__ == '__main__':
    main()
