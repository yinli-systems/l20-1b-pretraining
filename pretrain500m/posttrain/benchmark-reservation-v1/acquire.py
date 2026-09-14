"""Fetch hash-pinned public evaluation references; never execute benchmark code."""
import argparse
import ast
import concurrent.futures
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import time

import pyarrow.parquet as pq
import requests


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.next')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def get_file(spec, output):
    target = output / spec['provider'] / spec['repo'] / spec['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError('existing output needs inspection: ' + str(target))
    if not 0 < spec['bytes'] <= 8 * 1024**2:
        raise ValueError('source size exceeds acquisition bounds')
    with requests.get(spec['url'], timeout=(8, 30), stream=True) as response:
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(256 * 1024):
            size += len(chunk)
            if size > spec['bytes']:
                raise ValueError('download exceeds pinned source size')
            chunks.append(chunk)
    data = b''.join(chunks)
    if len(data) != spec['bytes']:
        raise ValueError('download size mismatch')
    sha256 = digest(data)
    git_blob_sha1 = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if spec['sha256']:
        if sha256 != spec['sha256']:
            raise ValueError('full LFS content digest mismatch')
    elif git_blob_sha1 != spec['git_blob_sha1']:
        raise ValueError('Git blob digest mismatch')
    with target.open('xb') as handle:
        handle.write(data)
    target.chmod(0o444)
    return dict(spec, local_path=str(target.resolve()), actual_sha256=sha256,
                actual_git_blob_sha1=git_blob_sha1, full_content_identity_verified=True)


def strings(value):
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def records(spec):
    path = Path(spec['local_path'])
    if spec['purpose'] != 'benchmark_reference_only':
        return
    if path.suffix == '.parquet':
        rows = pq.read_table(path).to_pylist()
    elif path.suffix == '.jsonl':
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix == '.tsv':
        data = list(csv.reader(io.StringIO(path.read_text()), delimiter='\t'))
        if len(data) != 250 or any(len(row) != 2 for row in data):
            raise ValueError('unexpected MGSM TSV shape')
        rows = [dict(question=row[0], answer=row[1]) for row in data]
    elif path.suffix == '.py':
        # Parse literals only; no imports, eval, exec or generated-code execution.
        tree = ast.parse(path.read_text())
        rows = [dict(exemplar=node.value, line=node.lineno) for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) >= 40]
    else:
        raise ValueError('unsupported reference format')
    field_map = {
        'openai/openai_humaneval': ['prompt', 'canonical_solution', 'test'],
        'google-research-datasets/mbpp': ['text', 'code', 'test_list', 'challenge_test_list', 'test_setup_code'],
        'google/IFEval': ['prompt'],
        'facebook/belebele': ['flores_passage', 'question', 'mc_answer1', 'mc_answer2', 'mc_answer3', 'mc_answer4'],
        'HuggingFaceH4/MATH-500': ['problem', 'solution', 'answer'],
        'google-research/url-nlp': ['exemplar'] if path.suffix == '.py' else ['question', 'answer'],
    }
    mandatory = {'openai/openai_humaneval': ['prompt', 'canonical_solution', 'test'],
                 'google-research-datasets/mbpp': ['code', 'test_list'],
                 'google/IFEval': ['prompt'],
                 'facebook/belebele': ['flores_passage', 'question'],
                 'HuggingFaceH4/MATH-500': ['problem', 'solution'],
                 'google-research/url-nlp': ['exemplar'] if path.suffix == '.py' else ['question', 'answer']}
    for ordinal, row in enumerate(rows):
        fields = []
        for key in mandatory[spec['repo']]:
            if key not in row or not list(strings(row[key])):
                raise ValueError('missing mandatory benchmark field: ' + spec['repo'] + '/' + key)
        # Sanitized MBPP uses prompt instead of text; keep the schema distinction.
        selected = field_map[spec['repo']]
        if spec['repo'] == 'google-research-datasets/mbpp':
            prompt_key = 'text' if 'text' in row else 'prompt'
            if prompt_key not in row or not list(strings(row[prompt_key])):
                raise ValueError('missing MBPP natural-language prompt')
            selected = [prompt_key] + [key for key in selected if key != 'text']
        for key in selected:
            for index, text in enumerate(strings(row.get(key))):
                fields.append(dict(name=key, index=index, text=text, sha256=digest(text.encode())))
        if not fields:
            raise ValueError('empty benchmark record')
        identity = dict(provider=spec['provider'], repo=spec['repo'], revision=spec['revision'],
                        path=spec['path'], row=ordinal, upstream_id=row.get('task_id', row.get('key')))
        yield dict(reference_id=digest(json.dumps(identity, sort_keys=True).encode()),
                   provenance=identity, domain=spec['domain'], fields=fields,
                   original_columns=sorted(row), original_row_sha256=digest(json.dumps(row, sort_keys=True,
                   ensure_ascii=False, default=str).encode()), source_sha256=spec['actual_sha256'], training_admitted=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inventory_bytes = args.inventory.read_bytes()
    inventory = json.loads(inventory_bytes)
    if sum(x['bytes'] for x in inventory['files']) != inventory['expected_total_bytes'] or inventory['expected_total_bytes'] > 64*1024**2:
        raise ValueError('inventory total mismatch or exceeds bound')
    args.output.mkdir(exist_ok=False)
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        fetched = list(pool.map(lambda item: get_file(item, args.output/'raw'), inventory['files']))
    write_json(args.output/'files.json', fetched)
    counts, field_counts = {}, {}
    with (args.output/'references.jsonl').open('x') as writer:
        for spec in fetched:
            key = spec['repo'] + ':' + spec['path']
            count = nfields = 0
            for row in records(spec):
                writer.write(json.dumps(row, ensure_ascii=False) + '\n')
                count += 1
                nfields += len(row['fields'])
            counts[key], field_counts[key] = count, nfields
    for spec in fetched:
        if digest(Path(spec['local_path']).read_bytes()) != spec['actual_sha256']:
            raise ValueError('reference source changed during extraction')
    if digest(args.inventory.read_bytes()) != digest(inventory_bytes):
        raise ValueError('inventory changed during acquisition')
    report = dict(status='REFERENCE_BUNDLE_READY_MATCHER_AND_CORPUS_SCAN_PENDING', training_admitted=False,
                  checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  files=len(fetched), bytes=sum(x['bytes'] for x in fetched), counts=counts, field_counts=field_counts,
                  records=sum(counts.values()), fields=sum(field_counts.values()), inventory_sha256=digest(inventory_bytes),
                  references_sha256=digest((args.output/'references.jsonl').read_bytes()),
                  file_receipts_sha256=digest((args.output/'files.json').read_bytes()), elapsed_seconds=time.time()-started,
                  scope='Supplemental benchmark references only. Legacy seven-task references remain required.',
                  limits=['No contamination matcher or training-corpus scan is supplied by acquisition.',
                          'No benchmark scoring, candidate selection or model quality measurement has occurred.',
                          'Benchmark code is parsed as data and is never executed.',
                          'Standalone short answer/choice fields require calibrated matching rules; do not exclude arbitrary training rows on common answers.',
                          'MATH-500 is a 500-problem subset, not coverage of all MATH; its pinned metadata did not declare a dataset license.',
                          'MGSM literal extraction is a reservation aid, not proof of original translated-problem family independence.'])
    write_json(args.output/'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
