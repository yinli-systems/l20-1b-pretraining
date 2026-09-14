"""Save immutable upstream metadata; no model weights or corpus download."""
from pathlib import Path
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent / 'sources'
MODELS = ['HuggingFaceTB/SmolLM2-360M', 'HuggingFaceTB/SmolLM-360M',
          'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen3-0.6B-Base',
          'facebook/MobileLLM-600M', 'apple/OpenELM-450M',
          'tiiuae/Falcon-H1-0.5B-Base',
          'allenai/DataDecide-fineweb-edu-530M',
          'allenai/DataDecide-dclm-baseline-530M',
          'allenai/DataDecide-dclm-baseline-qc-20p-530M']
MODELS.append('allenai/DataDecide-dclm-baseline-qc-7p-fw3-530M')


def get(url):
    with urllib.request.urlopen(url, timeout=45) as f:
        return f.read()


def save(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'Immutable source changed: {path}')
    else:
        path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def fetch(repo):
    out = ROOT / repo.replace('/', '--')
    out.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/api/models/{repo}?blobs=true'
    try:
        raw = get(url)
        info = json.loads(raw)
        record = {'repo': repo, 'revision': info['sha'], 'metadata_url': url,
                  'metadata_sha256': save(out / 'metadata.json', raw), 'files': []}
        for name in ['config.json', 'README.md', 'tokenizer_config.json',
                     'tokenizer.json', 'special_tokens_map.json']:
            url = f'https://huggingface.co/{repo}/resolve/{info["sha"]}/{name}'
            try:
                body = get(url)
                record['files'].append({'name': name, 'url': url, 'sha256': save(out / name, body)})
            except Exception as e:
                record['files'].append({'name': name, 'url': url, 'error': str(e)})
        print(repo,info['sha'],flush=True)
        return record
    except Exception as e:
        return {'repo': repo, 'error': str(e)}


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(5) as pool:
        records = list(pool.map(fetch, MODELS))
    (ROOT / 'model-inventory.json').write_text(json.dumps(records, indent=2)+'\n')
    url = 'https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu?blobs=true'
    raw = get(url)
    info = json.loads(raw)
    files = [x for x in info['siblings'] if x['rfilename'].startswith('sample/10BT/')]
    record = {'repo': 'HuggingFaceFW/fineweb-edu', 'revision': info['sha'],
              'metadata_url': url, 'metadata_sha256': hashlib.sha256(raw).hexdigest(),
              'files': files}
    (ROOT / 'fineweb-edu-10bt.json').write_text(json.dumps(record, indent=2)+'\n')
    for name in ['README.md']:
        body=get(f'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/{info["sha"]}/{name}')
        save(ROOT / ('fineweb-edu-'+name),body)
    print('CORPUS_REVISION',info['sha'],'FILES',len(files),flush=True)
