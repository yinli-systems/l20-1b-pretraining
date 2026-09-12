#!/usr/bin/env python3
"""Audit upstream DataDecide scores without treating OLMES as our lm-eval protocol."""
import ast
import concurrent.futures
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

REPO = 'allenai/DataDecide-eval-results'
REVISION = '9919b5a0e61e57a85021263918fa82d6ceaee038'
TASKS = {'hellaswag', 'arc_easy', 'arc_challenge', 'piqa', 'openbookqa', 'boolq', 'winogrande'}
CACHE = Path('cache/datadecide-upstream-20260912')
OUTPUT = Path('reports/metrics/datadecide-upstream-scan-20260912.json')


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            sha.update(block)
    return sha.hexdigest()


def download(item):
    path = CACHE / Path(item['rfilename']).name
    if not path.exists():
        with requests.get(f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{item["rfilename"]}', stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            with path.open('xb') as stream:
                for chunk in response.iter_content(4*1024**2):
                    stream.write(chunk)
    if path.stat().st_size != item['size'] or digest(path) != item['lfs']['sha256']:
        raise RuntimeError(f'Invalid source parquet: {path}; retained, not overwritten')
    print(f'VERIFIED {path}', flush=True)
    return path


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    CACHE.mkdir(parents=True, exist_ok=True)
    response = requests.get(f'https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true', timeout=30)
    response.raise_for_status()
    info = response.json()
    files = [x for x in info['siblings'] if x['rfilename'].startswith('data/train-')]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        paths = list(pool.map(download, files))
    rows = []
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            selected = batch.filter(pc.equal(batch.column('params'), '1B'))
            for row in selected.to_pylist():
                if row['task'] in TASKS and row['seed'] in ('default', 'large aux 2', 'large aux 3'):
                    row['primary_metric'] = float(ast.literal_eval(row.pop('metrics'))['primary_metric'])
                    rows.append(row)
    grouped = defaultdict(dict)
    for row in rows:
        key = (row['data'], row['step'], row['seed'], row['tokens'])
        if row['task'] in grouped[key]:
            raise ValueError(f'Duplicate result: {key}/{row["task"]}')
        grouped[key][row['task']] = row['primary_metric']
    complete = [dict(data=k[0], step=k[1], seed=k[2], tokens=k[3], scores=v,
                     seven_task_macro=sum(v.values()) / len(TASKS))
                for k,v in grouped.items() if set(v) == TASKS]
    final = defaultdict(list)
    for row in complete:
        if row['step'] == 69369:
            final[row['data']].append(row)
    ranking = []
    for name, values in final.items():
        seeds = {x['seed'] for x in values}
        if seeds != {'default', 'large aux 2', 'large aux 3'}:
            continue
        ranking.append({'recipe': name, 'mean_over_three_seeds': sum(x['seven_task_macro'] for x in values)/3,
                        'seed_rows': values})
    ranking.sort(key=lambda x: (-x['mean_over_three_seeds'], x['recipe']))
    output = {'revision': REVISION, 'source': f'https://huggingface.co/datasets/{REPO}/tree/{REVISION}',
              'source_files': files, 'script_sha256': digest(Path(__file__)), 'created_unix': time.time(),
              'selection_rule': 'Rank complete seven-task primary_metric means across default/large aux 2/large aux 3 at step69369; fixed before retrieval.',
              'ranking': ranking, 'available_1b_seven_task_rows': complete,
              'limitations': ['Upstream OLMES primary metrics are not this project lm_eval==0.4.9 zero-shot scores.',
                             'Recipe selection on public benchmark results is not independent held-out evaluation.',
                             'Published model-size definitions may exclude input embeddings; use measured total parameters in our compute proxy.']}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('x') as stream:
        json.dump(output, stream, indent=2)
    print(json.dumps({'ranking': [(x['recipe'], x['mean_over_three_seeds']) for x in ranking],
                      'complete_rows': len(complete), 'output': str(OUTPUT)}), flush=True)


if __name__ == '__main__':
    main()
