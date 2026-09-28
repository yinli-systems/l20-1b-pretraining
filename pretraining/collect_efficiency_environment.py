"""Freeze isolated adapter files and runtime provenance without touching CUDA."""
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from run_efficiency_evaluation import OUT, sha256


def tree_hashes(relative):
    root = OUT/relative
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f'Missing or linked adapter directory: {relative}')
    result = {}
    for path in sorted(root.rglob('*')):
        if '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        if path.is_symlink():
            raise ValueError(f'Linked adapter artifact: {path}')
        if path.is_file():
            result[str(path.relative_to(OUT))] = sha256(path)
    return result


def main():
    destination = OUT/'adapter-environment.json'
    if destination.exists():
        raise FileExistsError(destination)
    admissions = {}
    for kind, name in (('author_hf_olmo','olmo-cpu-smoke.json'),
                       ('author_open_lm','openlm-cpu-smoke.json'),
                       ('transformers_builtin','pythia-cpu-smoke.json')):
        path = OUT/name
        record = json.loads(path.read_text())
        expected = 'actual_checkpoint_cpu_pass' if kind == 'transformers_builtin' else 'cpu_toy_pass'
        if record['status'] != expected or record['device'] != 'cpu':
            raise ValueError(f'Adapter smoke failed: {kind}')
        admissions[kind] = {'receipt':name,'sha256':sha256(path),'result':record,
                            'scope':'CPU only; each real checkpoint still requires strict loading and finite GPU logits.'}
    adapters = {'author_hf_olmo':tree_hashes('vendor'),
                'author_open_lm':{**tree_hashes('open-lm-author-clean'),
                                  **tree_hashes('openlm-deps/einops')}}
    system = Path('/home/hhai/pretrain/.venv/lib/python3.12/site-packages')
    sources = {}
    for module in ('lm_eval','transformers'):
        for path in sorted((system/module).rglob('*.py')):
            sources[str(path.relative_to(system))] = sha256(path)
        if module == 'lm_eval':
            for path in sorted((system/module).rglob('*.yaml')):
                sources[str(path.relative_to(system))] = sha256(path)
    payload = {'created_unix':time.time(),'python':platform.python_version(),
               'package_versions':{p:importlib.metadata.version(p) for p in
                                   ('torch','transformers','numpy','scipy','datasets','lm_eval','tokenizers','accelerate')},
               'adapter_admission':admissions,'adapter_file_sha256':adapters,
               'system_source_root':str(system),'system_source_sha256':sources,
               'openlm_author_commit':'9bb92ef1689333534b7057942a20d18a46d1fa52',
               'openlm_patch':'Only defer the two xformers imports to their xformers-only call sites. Admitted checkpoints explicitly use torch_attn and swiglu_torch.',
               'training_environment_modified':False}
    with destination.open('x') as stream:
        json.dump(payload,stream,indent=2)
    print(json.dumps({'path':str(destination),'sha256':sha256(destination),
                      'adapter_files':{k:len(v) for k,v in adapters.items()},
                      'system_source_files':len(sources)}),flush=True)


if __name__ == '__main__':
    main()
