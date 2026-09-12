"""Exclude two additional context-free/copypasta samples found on expanded audit.

Rerun from the original intact refill to avoid compounding alignment tail loss.
All v1 candidate and review files remain immutable. No change of token mixture.
"""
import json
from pathlib import Path

from continuation_common import RUN, atomic_json, lock, sha256
from curate_continuation import curate
from refine_continuation_b_review import reject_reason as previous_reject
from prepare_continuation_b_verified import verify_inputs

VERSION = '20260911-B-reviewed-v2'
MORE = {'<urn:uuid:5fa09a32-b384-4943-bb02-3565a90ba44c>':'threat_copypasta_comment_fragment',
        '<urn:uuid:cacd099b-799a-4579-a590-2e34e219084e>':'context_free_event_fragment'}


def reject_reason(source,identifier,text,tokens):
    return MORE.get(identifier) or previous_reject(source,identifier,text,tokens)


def main():
    handle=lock(RUN/'prepare.lock')
    before=verify_inputs()
    destination=RUN/'b-reviewed-v2'
    raw=RUN/'b-candidate-v1/new-dclm'
    curate('dclm',input_path=raw/'manifest.json',input_inventory=raw/'documents.sqlite',
           directory=destination/'new-dclm',quality_filter=reject_reason)
    for source in ('web','math','code','narrative','dclm'):
        previous=RUN/f'b-reviewed-v1/{source}/manifest.json'
        m=json.loads(previous.read_text())
        if source=='dclm':
            paths=[RUN/'refined-v3/dclm/manifest.json',destination/'new-dclm/manifest.json']
            parts=[json.loads(p.read_text()) for p in paths]
            m['shards']=[x for part in parts for x in part['shards']]
            m['tokens']=sum(x['tokens'] for x in m['shards'])
            m['input_manifests']=[{'path':str(p),'sha256':sha256(p)} for p in paths]
        if m['tokens']<m['required_input_tokens']:
            raise RuntimeError('expanded audit leaves a shortfall; no relaxed quota')
        for item in m['shards']:
            if sha256(Path(item['path']))!=item['sha256']:
                raise RuntimeError('final B shard changed')
        out=destination/source/'manifest.json'
        if out.exists() and json.loads(out.read_text())!=m:
            raise RuntimeError('final B manifest is immutable')
        atomic_json(out,m)
    after=verify_inputs()
    if before!=after:
        raise RuntimeError('exclusion inputs changed during final review')
    atomic_json(destination/'input-verification.json',{'stage':'pre_and_post_checks_passed','inputs':after})
    handle.close()
    print('FINAL_B_POOLS_READY_PENDING_ADMISSION',flush=True)


if __name__=='__main__':
    main()
