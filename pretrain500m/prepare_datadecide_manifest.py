"""Freeze a bounded, hash-addressed QC7/FW3 token shard subset."""
import json
import urllib.parse
import urllib.request
from pathlib import Path

REPO='allenai/DataDecide-data-recipes'
PREFIX='preprocessed/dclm/v0_rep32_ft7percentile_fw3/gpt-neox-olmo-dolma-v1_5'
metadata=json.loads((Path(__file__).resolve().parent/'sources/datadecide-data-recipes.json').read_text())
revision=metadata['sha']
url=f'https://huggingface.co/api/datasets/{REPO}/tree/{revision}/{PREFIX}?limit=1000'
with urllib.request.urlopen(url,timeout=60) as response:
    files=json.load(response)
files=sorted((x for x in files if x['type']=='file' and x['path'].endswith('.npy')),key=lambda x:x['path'])
assert len(files)==256
# Files are raw little-endian uint16 token IDs despite the historic .npy suffix.
# Select the shortest lexical prefix containing at least the predeclared 18B cap.
selected=[]
for item in files:
    selected.append(item)
    if sum(x['size']//2 for x in selected)>=18_000_000_000:
        break
assert all(x.get('lfs',{}).get('oid') and x['size'] for x in selected)
manifest={'version':1,'repo':REPO,'revision':revision,'prefix':PREFIX,'tree_url':url,
          'storage_format':'raw little-endian uint16 token IDs; verified by upstream OLMo mapping and range probe',
          'selection_rule':f'lexicographically first {len(selected)} of 256 shards, shortest prefix covering 18B raw tokens; frozen before training',
          'target_tokens':16_000_000_000,'maximum_tokens':18_000_000_000,
          'files':[{'path':x['path'],'bytes':x['size'],'sha256':x['lfs']['oid'],
                    'url':f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/'+urllib.parse.quote(x['path'])}
                   for x in selected],
          'total_download_bytes':sum(x['size'] for x in selected),
          'raw_token_count':sum(x['size']//2 for x in selected),
          'limitations':['Numpy headers determine the exact usable token count after download.',
                         'Recipe selection used public upstream evaluation and is not an independent quality result.',
                         'Corpus-level benchmark decontamination is a separate formal-run gate.']}
out=Path(__file__).resolve().parent/'sources/datadecide-qc7-fw3-16b-manifest.json'
out.write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'files':len(selected),'bytes':manifest['total_download_bytes'],'output':str(out)}))
