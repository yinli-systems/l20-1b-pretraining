
from pathlib import Path
import datetime,gzip,hashlib,json,re,unicodedata
r=Path('/ssd/scxi253/pretrain500m-20260912-v1');audit=r/'data/contamination-v1-three-tranche';out=r/'receipts/supplemental-exact-text-exclusions-v2.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((audit/'report.json').read_text());cf=audit/'candidate-exclusions.jsonl';refs=r/'evaluation/benchmark-reservation-v1/data/references.jsonl'
assert sha(cf)==report['candidates_sha256'] and sha(refs)==report['references_sha256'] and not out.exists()
reference={}
with refs.open() as f:
 for line in f:
  q=json.loads(line);reference[q['reference_id']]=q
candidates=[json.loads(l) for l in cf.read_text().splitlines()];assert len(candidates)==report['counts']['candidate_exclusion_rows']==3
verified=[];raw_bindings={}
def norm(t,mode):
 t=unicodedata.normalize('NFC',t)
 if mode=='prose':t=unicodedata.normalize('NFC',t.casefold())
 return re.sub(r'\s+',' ',t).strip()
for c in candidates:
 root=Path(report['tranches'][c['tranche']]['path']);assert root==r/('data/diverse-intake-v'+str(c['tranche']+1))
 raw=root/(c['source_id']+'.jsonl.gz');assert sha(raw)==report['input_bindings'][str(raw)];raw_bindings[raw]=sha(raw)
 with gzip.open(raw,'rt') as f:
  row=next(json.loads(line) for i,line in enumerate(f) if i==c['row'])
 text=row.get('text') or row.get('content');assert hashlib.sha256(text.encode()).hexdigest()==c['text_sha256'] and row['_provenance']==c['provenance']
 assert c['matches'] and not c['saved_matches_truncated']
 for hit in c['matches']:
  normalized=norm(text,hit['mode']);span=normalized[hit['normalized_start']:hit['normalized_end']]
  assert hashlib.sha256(span.encode()).hexdigest()==hit['matched_sha256']
  assert hit['reference_owners']
  for owner in hit['reference_owners']:
   ref=reference[owner['reference_id']];assert ref['provenance']==owner['provenance']
   assert hit['mode']==('literal' if ref['domain'] in ['code','math','multilingual_math'] else 'prose')
   fields=[f for f in ref['fields'] if f['name']==owner['field'] and f['index']==owner['field_index']];assert len(fields)==1
   field=fields[0];assert hashlib.sha256(field['text'].encode()).hexdigest()==field['sha256']
   assert span in norm(field['text'],hit['mode'])
 verified.append({**{k:c[k] for k in ['source_id','tranche','row','text_sha256','provenance','matches']},'url':row.get('url'),'repo_name':row.get('repo_name')})
old=json.loads((r/'receipts/supplemental-exact-text-exclusions-v1.json').read_text());hashes=sorted({c['text_sha256'] for c in verified});assert set(old['exclude_text_sha256_across_all_tranches'])<=set(hashes)
assert all(sha(p)==h for p,h in raw_bindings.items()) and sha(refs)==report['references_sha256'] and sha(cf)==report['candidates_sha256']
result={'status':'EXACT_TEXT_EXCLUSION_PLAN_BOUND_AND_VERIFIED','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'audit_report_sha256':sha(audit/'report.json'),'candidate_file_sha256':sha(cf),'reference_bundle_sha256':sha(refs),'exclude_text_sha256_across_all_tranches':hashes,'verified_candidates':verified,'candidate_rows':len(verified),'training_admitted':False,'raw_files_modified':False,'packing_has_consumed_plan':False,'family_or_near_duplicate_expansion_completed':False,'supersedes':str(r/'receipts/supplemental-exact-text-exclusions-v1.json'),'rule':'Exclude byte-identical content matching these hashes from every training pool before packing; apply broader family/near-duplicate rules separately.'}
with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
