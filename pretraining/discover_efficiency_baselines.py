#!/usr/bin/env python3
"""Resolve public baseline metadata to immutable revisions; never download weights."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests

METADATA_FILES = {
    'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
    'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt',
    'tokenizer.model', 'model.safetensors.index.json', 'pytorch_model.bin.index.json',
}
DATADECIDE_RECIPES = ('dclm-baseline-qc-20p', 'fineweb-edu', 'dclm-baseline', 'dclm-baseline-qc-7p-fw3')
DATADECIDE_STEPS = (10000, 12500, 15000, 20000, 42500)


def hash_bytes(content):
    return hashlib.sha256(content).hexdigest()


def fetch(url, cache):
    key = hashlib.sha256(url.encode()).hexdigest()
    path = cache / (key + '.json')
    if path.exists():
        return json.loads(path.read_text())
    response = requests.get(url, timeout=(10, 40))
    record = {'url': url, 'status': response.status_code, 'time': time.time(),
              'body_sha256': hash_bytes(response.content), 'text': response.text}
    with path.open('x') as stream:
        json.dump(record, stream)
    return record


def get_json(url, cache):
    record = fetch(url, cache)
    if record['status'] != 200:
        raise RuntimeError(f"HTTP {record['status']} at {url}")
    return json.loads(record['text'])


def inference_files(siblings):
    root = [x for x in siblings if '/' not in x['rfilename']]
    safe = [x for x in root if x['rfilename'].endswith('.safetensors')]
    weights = safe or [x for x in root if x['rfilename'] == 'pytorch_model.bin'
                       or re.fullmatch(r'pytorch_model-\d+-of-\d+\.bin', x['rfilename'])]
    if not weights:
        raise ValueError('No supported root-level model weights')
    selected = weights + [x for x in root if x['rfilename'] in METADATA_FILES]
    output = []
    for item in selected:
        output.append({'path': item['rfilename'], 'size': item['size'],
                       'sha256': item.get('lfs', {}).get('sha256'),
                       'git_blob_sha1': item['blobId']})
    return sorted(output, key=lambda x: x['path'])


def resolve(job, cache):
    result = dict(job)
    try:
        repo = job['repo']
        ref = job.get('ref', 'main')
        url = f'https://huggingface.co/api/models/{repo}/revision/{quote(ref, safe="")}?blobs=true'
        info = get_json(url, cache)
        revision = info['sha']
        assert re.fullmatch('[0-9a-f]{40}', revision)
        config_url = f'https://huggingface.co/{repo}/resolve/{revision}/config.json'
        config_record = fetch(config_url, cache)
        if config_record['status'] != 200:
            raise RuntimeError(f"config HTTP {config_record['status']}")
        config = json.loads(config_record['text'])
        files = inference_files(info['siblings'])
        card_record = fetch(f'https://huggingface.co/{repo}/resolve/{revision}/README.md', cache)
        result.update({
            'resolution_status': 'resolved', 'revision': revision, 'config': config,
            'config_sha256': config_record['body_sha256'], 'gated': info.get('gated'),
            'files': files, 'download_bytes': sum(x['size'] for x in files),
            'parameter_metadata': info.get('safetensors'), 'license': info.get('cardData', {}).get('license'),
            'metadata_url': url, 'metadata_sha256': hash_bytes(json.dumps(info, sort_keys=True).encode()),
            'model_card_url': card_record['url'], 'model_card_sha256': card_record['body_sha256'],
            'model_card_status': card_record['status'],
        })
        # Adapter admission is separate: a resolved repository is not yet runnable.
        kind = config.get('model_type')
        result['adapter'] = {'hf_olmo': 'author_hf_olmo', 'openlm': 'author_open_lm',
                             'mobilellm': 'author_mobilellm'}.get(kind, 'transformers_builtin')
    except Exception as exc:
        result.update({'resolution_status': 'blocked', 'error': str(exc)})
    return result


def candidate_jobs(cache):
    jobs = [
        {'id': 'cerebras_1_3b', 'repo': 'cerebras/Cerebras-GPT-1.3B',
         'training_tokens': 26_300_000_000, 'token_kind': 'paper_rounded', 'category': 'natural_corpus_base'},
    ]
    # Predeclared anchors, NOT a claim that these are the best of all 25 recipes.
    # Both neighbors of 20B are retained, avoiding selective nearest-point choice.
    for recipe in DATADECIDE_RECIPES:
        repo = f'allenai/DataDecide-{recipe}-1B'
        refs = get_json(f'https://huggingface.co/api/models/{repo}/refs', cache)
        branches = {x['name']: x['targetCommit'] for x in refs['branches']}
        for step in DATADECIDE_STEPS:
            ref = f'step{step}-seed-default'
            jobs.append({'id': f'dd_{recipe.replace("-", "_")}_{step}', 'repo': repo, 'ref': ref,
                         'ref_exists': ref in branches, 'training_tokens': step * 704 * 2048,
                         'token_kind': 'step_times_global_batch_times_sequence_length',
                         'token_source': 'https://github.com/allenai/DataDecide#hyperparameters',
                         'category': 'natural_corpus_base', 'seed': 'default',
                         'selection': ('Top three-seed seven-task upstream OLMES mean at step69369 among 25 recipes; not held-out model selection.'
                                       if recipe == 'dclm-baseline-qc-20p' else 'predeclared recipe anchor')})
    for name, repo in [
        ('weborganizer_dclm', 'WebOrganizer/LM-1b_1x-DCLMFasttext'),
        ('weborganizer_domain_mix', 'WebOrganizer/LM-1b_1x-DCLMFasttext_over_Topics_x_Formats_for_MMLU_and_Hellaswag'),
    ]:
        jobs.append({'id': name, 'repo': repo, 'training_tokens': 28_795_904_000,
                     'token_kind': 'author_training_config', 'category': 'natural_corpus_base',
                     'selection': 'domain-mix recipe optimized for MMLU and HellaSwag' if name.endswith('mix') else 'global DCLM score filter'})
    for name, repo, tokens, category in [
        ('phi_1_5', 'microsoft/phi-1_5', 150_000_000_000, 'teacher_synthetic_reference'),
        ('opt_1_3b', 'facebook/opt-1.3b', 180_000_000_000, 'natural_corpus_base'),
        ('pythia_1b', 'EleutherAI/pythia-1b', 300_000_000_000, 'natural_corpus_base'),
        ('falcon_rw_1b', 'tiiuae/falcon-rw-1b', 350_000_000_000, 'natural_corpus_base'),
        ('mobilellm_1b', 'facebook/MobileLLM-1B', 1_000_000_000_000, 'natural_corpus_base'),
    ]:
        jobs.append({'id': name, 'repo': repo, 'training_tokens': tokens,
                     'token_kind': 'author_rounded', 'category': category})
    # Discover exact official model names rather than guessing IDs from a paper table.
    # Early checkpoints live on branches of a different, official collection.
    # Keep the author's nominal labels; do not rename the 105B branch as 103B.
    for step, nominal_b in ((5,10),(10,21),(15,31),(30,63),(50,105)):
        jobs.append({'id':f'tinyllama_early_{nominal_b}b',
                     'repo':'TinyLlama/tinyLlama-intermediate-checkpoints',
                     'ref':f'step-{step}k-token-{nominal_b}B',
                     'training_tokens':nominal_b*1_000_000_000,
                     'token_kind':'nominal_official_branch_label',
                     'token_source':'https://huggingface.co/TinyLlama/tinyLlama-intermediate-checkpoints/refs',
                     'category':'natural_corpus_base',
                     'selection':'predeclared early trajectory milestones; labels are rounded, not exact token counters'})
    tiny = get_json('https://huggingface.co/api/models?author=TinyLlama&search=intermediate&limit=100', cache)
    for item in tiny:
        repo = item['id']
        if not re.fullmatch(r'TinyLlama/TinyLlama-1\.1B-intermediate-step-\d+k-(?:token-)?[\d.]+[bBtT]', repo):
            continue
        nominal = re.search(r'-([\d.]+)([bBtT])$', repo)
        tokens = int(float(nominal[1]) * (1e12 if nominal[2].lower() == 't' else 1e9))
        jobs.append({'id': 'tinyllama_' + repo.rsplit('-', 1)[1].lower().replace('.', '_'),
                     'repo': repo, 'training_tokens': tokens, 'token_kind': 'nominal_repo_label',
                     'category': 'natural_corpus_base', 'reuse_existing_if_verified': tokens == 3_000_000_000_000})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('cache/efficiency-baseline-metadata-20260911'))
    parser.add_argument('--output', type=Path, default=Path('reports/receipts/efficiency-baseline-inventory-20260911.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.cache.mkdir(parents=True, exist_ok=True)
    work = candidate_jobs(args.cache)
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        jobs = list(pool.map(lambda job: resolve(job, args.cache), work))
    payload = {'created_unix': time.time(), 'discovery_script_sha256': hash_bytes(Path(__file__).read_bytes()),
               'jobs': jobs, 'unresolved_requests': [
                   {'id': 'meta_lingua_1b_60b', 'status': 'checkpoint_not_located', 'source': 'https://github.com/facebookresearch/lingua', 'note': 'Author README reports scores but no matching released checkpoint was found; do not replace by a similarly named model.'},
                   {'id': 'official_dclm_1b_1x', 'status': 'checkpoint_not_located', 'source': 'https://www.datacomp.ai/dclm/leaderboard.html', 'note': 'WebOrganizer DCLM-filtered models are author-released research baselines, not the original DCLM competition checkpoint.'},
               ],
               'claim_boundary': 'No ranking or Pareto result. Public metadata resolution is not weight verification, adapter admission or completed evaluation.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(payload, stream, indent=2)
    for job in jobs:
        print(json.dumps({k: job.get(k) for k in ['id','resolution_status','revision','adapter','download_bytes','error']}), flush=True)
    print(f'INVENTORY {len(jobs)} candidates -> {args.output}')


if __name__ == '__main__':
    main()
