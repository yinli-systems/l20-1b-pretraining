"""Collect public, revision-pinned cards and configs; never fetch gated payloads.

This inventory is research evidence, not a training-data admission receipt.
Uses no HF token, never accepts agreements, and executes no dataset code.
"""
import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
DATASETS = [
    'HuggingFaceFW/fineweb-edu',
    'mlfoundations/dclm-baseline-1.0',
    'mlfoundations/dclm-baseline-1.0-parquet',
    'HuggingFaceFW/finepdfs-edu',
    'HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT',
    'HuggingFaceTB/finemath', 'HuggingFaceTB/stack-edu',
    'HuggingFaceTB/cosmopedia-v2', 'epfml/FineWeb2-HQ',
    'HuggingFaceFW/fineweb-2', 'nvidia/Nemotron-CC-v2.1',
    'nvidia/Nemotron-CC-Math-v1', 'nvidia/Nemotron-CC-Code-v1',
    'nvidia/Nemotron-Pretraining-Specialized-v1',
    'allenai/dolma3_dolmino_mix-10B-1025', 'allenai/dolmino-mix-1124',
    'LLM360/MegaMath', 'HuggingFaceTB/smol-smoltalk',
    'HuggingFaceTB/smoltalk2', 'open-thoughts/OpenThoughts3-1.2M',
    'allenai/Dolci-Instruct-SFT', 'nvidia/OpenMathInstruct-2',
    'HuggingFaceTB/smollm-corpus', 'HuggingFaceFW/finewiki',
    'HuggingFaceFW/dclm_100BT', 'nvidia/Nemotron-SFT-Math-v3',
    'nvidia/Open-SWE-Traces', 'nvidia/Nemotron-Instruction-Following-Chat-v1',
]
MODELS = ['Qwen/Qwen3-0.6B-Base', 'Qwen/Qwen3.5-0.8B',
          'LiquidAI/LFM2.5-350M', 'LiquidAI/LFM2.5-1.2B-Instruct',
          'facebook/MobileLLM-R1-360M-base', 'HuggingFaceTB/SmolLM2-360M',
          'google/gemma-3-270m', 'tiiuae/Falcon-H1-0.5B-Base']

def collect(kind, repo):
    record = {'kind': kind, 'repo_id': repo,
              'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    folder = ROOT / 'sources' / repo.replace('/', '--')
    folder.mkdir(parents=True, exist_ok=True)
    try:
        res = requests.get(f'https://huggingface.co/api/{kind}/{repo}', timeout=35)
        record['api_status'] = res.status_code
        res.raise_for_status()
        meta = res.json()
        record['canonical_repo_id'] = meta.get('id')
        for key in ['sha','gated','private','lastModified','createdAt','downloads']:
            record[key] = meta.get(key)
        card = meta.get('cardData') or {}
        record['declared_license'] = card.get('license')
        record['card_data'] = card
        record['files'] = [x['rfilename'] for x in meta.get('siblings', [])]
        prefix = 'datasets/' if kind == 'datasets' else ''
        record['card_url'] = f'https://huggingface.co/{prefix}{repo}/blob/{meta["sha"]}/README.md'
        record['snapshots'] = {}
        for name in (['README.md'] if kind == 'datasets' else ['README.md','config.json']):
            url = f'https://huggingface.co/{prefix}{repo}/resolve/{meta["sha"]}/{name}'
            doc = requests.get(url, timeout=35)
            entry = {'url':url,'http_status':doc.status_code}
            if doc.ok:
                payload = doc.content
                (folder / name).write_bytes(payload)
                entry.update(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
            record['snapshots'][name] = entry
        record['payload_admission'] = 'GATED_EXCLUDED' if meta.get('gated') else 'NOT_AUDITED'
    except Exception as exc:
        record['error'] = str(exc).split('?')[0][:400]
    (folder / 'metadata.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
    return record

def main():
    targets = [('datasets',x) for x in DATASETS] + [('models',x) for x in MODELS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        rows = list(pool.map(lambda x:collect(*x),targets))
    (ROOT / 'source-inventory.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False)+'\n')
    for row in rows:
        print(json.dumps({k:row.get(k) for k in ['repo_id','api_status','sha','gated','declared_license','error']}),flush=True)

if __name__ == '__main__':
    main()
