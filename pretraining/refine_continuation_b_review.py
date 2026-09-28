"""Post-excerpt-review exclusions. Never alter the candidate or A manifests."""
import json
from pathlib import Path

from continuation_common import RUN, BLOCK, GLOBAL, PILOT_STEPS, MIXES, quotas, lock, atomic_json, sha256
from continuation_narrative_quality import reject_reason as narrative_reject
from continuation_general_quality import reject_reason as general_reject
from curate_continuation import curate
from prepare_continuation_b_verified import verify_inputs

VERSION = '20260911-B-reviewed-v1'
DEST = RUN/'b-reviewed-v1'
EXCLUDED = {
    '<urn:uuid:7e9446cc-79fd-436c-ae0e-f48d1999d8e2>':'meme_page_dominated_by_social_ui_and_low_context_replies',
    '<urn:uuid:617b45e6-c06a-4ff4-ae07-b2ba062c3dab>':'observed_inaccurate_historical_summary',
    '<urn:uuid:9081ed9f-b591-4e83-8f36-1602d9dfd82e>':'local_provider_promotional_copy',
}


def reject_reason(source,identifier,text,tokens):
    if identifier in EXCLUDED:
        return EXCLUDED[identifier]
    if source=='narrative':
        if '[missing text]' in text[:5000].lower():
            return 'declared_missing_source_text'
        if sum(128<=ord(c)<160 for c in text)>10:
            return 'repeated_c1_encoding_artifacts'
        return narrative_reject(source,identifier,text,tokens)
    return general_reject(source,identifier,text,tokens)


def main():
    handle = lock(RUN/'prepare.lock')
    before = verify_inputs()
    raw = RUN/'b-candidate-v1'
    for source in ('narrative','dclm'):
        directory = raw/f'new-{source}'
        curate(source,input_path=directory/'manifest.json',directory=DEST/f'new-{source}',
               input_inventory=directory/'documents.sqlite',quality_filter=reject_reason)
    q = quotas(MIXES['B'],GLOBAL*PILOT_STEPS)
    for source in ('web','dclm','math','code','narrative'):
        if source=='narrative':
            paths=[DEST/'new-narrative/manifest.json']
        elif source=='dclm':
            paths=[RUN/'refined-v3/dclm/manifest.json',DEST/'new-dclm/manifest.json']
        else:
            paths=[raw/source/'manifest.json']
        parts=[json.loads(p.read_text()) for p in paths]
        records=[x for m in parts for x in m['shards']]
        required=q['fresh_'+source]*BLOCK
        total=sum(x['tokens'] for x in records)
        if total<required:
            raise RuntimeError(f'reviewed {source} below quota: {total} < {required}')
        for record in records:
            if sha256(Path(record['path']))!=record['sha256']:
                raise RuntimeError('reviewed shard changed')
        m={k:parts[0][k] for k in ('source','base_inputs_sha256','benchmark_sha256','tokenizer_sha256')}
        m.update(tokens=total,shards=records,required_input_tokens=required,
                 input_manifests=[{'path':str(p),'sha256':sha256(p)} for p in paths],
                 stage='post_excerpt_review_candidate',training_enabled=False)
        out=DEST/source/'manifest.json'
        if out.exists() and json.loads(out.read_text())!=m:
            raise RuntimeError('frozen reviewed manifest differs')
        atomic_json(out,m)
        print(json.dumps({'source':source,'reviewed_tokens':total,'required':required}),flush=True)
    after=verify_inputs()
    if before!=after:
        raise RuntimeError('exclusion inputs changed during review curation')
    atomic_json(DEST/'input-verification.json',{'stage':'pre_and_post_checks_passed','inputs':after})
    handle.close()


if __name__=='__main__':
    main()
