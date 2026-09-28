#!/usr/bin/env python3
"""Build the public language comparison from the unchanged frozen evidence.

No new inference, task selection, metric changes, or model-ranking search.
--check verifies generated tables and figures without requiring a GPU.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'pretraining/reports/metrics/efficiency-all-results-final-20260912.json'
DEST = 'results/language-model'
START = '<!-- language-results:start -->'
END = '<!-- language-results:end -->'
SELECTED = (
    'dd_dclm_baseline_qc_7p_fw3_12500', 'tinyllama_early_21b',
    'weborganizer_domain_mix', 'phi_1_5', 'opt_1_3b', 'pythia_1b',
    'falcon_rw_1b', 'tinyllama_1t', 'tinyllama_1_5t', 'tinyllama_2t', 'tinyllama_2_5t',
)
NAMES = {
    'pythia_1b': 'Pythia-1B', 'opt_1_3b': 'OPT-1.3B',
    'falcon_rw_1b': 'Falcon-RW-1B', 'phi_1_5': 'Phi-1.5 (synthetic-data reference)',
    'weborganizer_domain_mix': 'WebOrganizer · domain mix',
    'weborganizer_dclm': 'WebOrganizer · DCLM',
    'dd_dclm_baseline_qc_7p_fw3_12500': 'DataDecide · QC7/FW3 · 18B',
}

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_source(root: Path = ROOT) -> dict:
    path = root / SOURCE
    binding = json.loads((root/'pretraining/reproducibility/manifest.json').read_text())
    expected = binding['bound_files']['reports/metrics/efficiency-all-results-final-20260912.json']
    if digest(path) != expected:
        raise ValueError('Frozen language evidence hash mismatch')
    source = json.loads(path.read_text())
    if len(source['checkpoints']) != 36 or len({x['job'] for x in source['checkpoints']}) != 36:
        raise ValueError('Expected all 36 unique declared baselines')
    for row in source['checkpoints']:
        for field in ('seven_task_macro', 'six_task_without_boolq'):
            metric = row[field]
            delta = 100*(metric['ours']-metric['baseline'])
            if not math.isclose(delta, metric['ours_minus_baseline_pp'], abs_tol=1e-11):
                raise ValueError('Stored comparison delta does not recompute')
    return source

def label(job: str) -> str:
    if job in NAMES:
        return NAMES[job]
    if job.startswith('tinyllama_early_'):
        return 'TinyLlama · '+job.split('_')[-1].upper()
    if job.startswith('tinyllama_'):
        return 'TinyLlama · '+job[len('tinyllama_'):].replace('_','.').upper()
    return job

def tokens(value: int) -> str:
    return f'{value/1e12:g}T' if value >= 1e12 else f'{value/1e9:.2f}'.rstrip('0').rstrip('.')+'B'

def row_text(record: dict) -> str:
    return (f"| {label(record['job'])} | {record['total_parameters']/1e9:.2f}B | "
            f"{tokens(record['training_tokens'])} | {100*record['seven_task_macro']['baseline']:.2f} | "
            f"{100*record['six_task_without_boolq']['baseline']:.2f} |")

def main_table(source: dict) -> str:
    ours = source['ours']; lookup = {x['job']:x for x in source['checkpoints']}
    lines = ['| Model / checkpoint | Parameters | Training tokens | 7-task ↑ | 6-task ↑ |',
             '|---|---:|---:|---:|---:|',
             f"| **L20-1B · ours** | **{ours['parameters']/1e9:.2f}B** | **20B** | **{ours['seven_task_macro_percent']:.2f}** | **{ours['six_task_without_boolq_percent']:.2f}** |"]
    lines += [row_text(lookup[job]) for job in SELECTED]
    return '\n'.join(lines)

def payload(source: dict, root: Path = ROOT) -> dict:
    fields = ('index','job','repo','revision','category','training_tokens','token_kind',
              'total_parameters','training_compute_proxy_ratio_vs_ours_6ND',
              'seven_task_macro','six_task_without_boolq')
    return {'schema':'language-showcase-v1','presentation_date':'2026-09-28',
            'study_frozen_date':'2026-09-12','source':SOURCE,'source_sha256':digest(root/SOURCE),
            'ours':source['ours'],'protocol':source['protocol'],
            'homepage_selection':list(SELECTED),
            'selection_rule':'Recognizable families, TinyLlama training trajectory, a near-budget and a stronger modern data-recipe reference; all 36 remain in the figure and full table.',
            'checkpoints':[{k:row[k] for k in fields} for row in source['checkpoints']],
            'new_inference_performed':False,
            'claims':{'different_training_recipes':True,'different_tokenizers':True,
                      'evaluation_sample_uncertainty_not_training_seeds':True,
                      'multiplicity_adjusted':False,'confidence_interval_crossing_zero_is_not_equivalence':True,
                      'six_task_sensitivity_excludes_not_explicitly_decontaminated_BoolQ':True,
                      '6ND_is_not_measured_hardware_cost':True,'global_SOTA_claim':False}}

def report(source: dict) -> str:
    lookup = {x['job']:x for x in source['checkpoints']}
    text = '''# Language-model comparison

**1.100B parameters · 19.9997B prediction tokens · one NVIDIA L20**

This presentation is generated from the unchanged September 12, 2026 frozen
study. It adds no new model inference or leaderboard claims.

## At a glance

'''+main_table(source)+'''

## Paired comparisons that support the headline

| Baseline | L20 minus baseline, pp | Paired 95% interval, pp | Interpretation |
|---|---:|---:|---|
'''
    for job in ['tinyllama_early_21b','pythia_1b','opt_1_3b','tinyllama_1t','tinyllama_1_5t','tinyllama_2t','tinyllama_2_5t','dd_dclm_baseline_qc_7p_fw3_12500','phi_1_5']:
        m = lookup[job]['seven_task_macro']; lo,hi=m['paired_95_ci_pp']
        interpretation = 'Higher in this protocol' if lo>0 else 'Lower in this protocol' if hi<0 else 'Interval spans zero; not proof of equivalence'
        text += f"| {label(job)} | {m['ours_minus_baseline_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {interpretation} |\n"
    text += '''
The 20B-versus-1T comparison is a **50× difference in reported token exposure**,
not proof of 50× faster training, 50× lower total cost, or equal general capability.
The model also loses to stronger candidates; these are included below.

## All 36 frozen baselines

Training compute is a **6ND approximation**, relative to L20-1B's 20B run.
It excludes data curation/classifier/teacher and experiment-search compute.
Parameter counts are measured totals, which can differ from rounded model names.

| Checkpoint | Parameters | Training tokens | 6ND / ours | 7-task | 6-task | L20−baseline 7-task, pp | Paired 95% interval, pp |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for row in source['checkpoints']:
        m=row['seven_task_macro'];lo,hi=m['paired_95_ci_pp']
        text += f"| `{row['job']}` | {row['total_parameters']/1e9:.4f}B | {tokens(row['training_tokens'])} | {row['training_compute_proxy_ratio_vs_ours_6ND']:.3f}× | {100*m['baseline']:.4f} | {100*row['six_task_without_boolq']['baseline']:.4f} | {m['ours_minus_baseline_pp']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |\n"
    text += '''
## Task-level view of the same comparison

No task is newly selected for this display. The seven-task suite was already
fixed: HellaSwag, PIQA, WinoGrande, OpenBookQA, ARC-Easy, ARC-Challenge and BoolQ.
Six-task sensitivity removes BoolQ, which was not explicitly decontaminated
from the original corpus. Task-specific metrics are the original acc/acc_norm.

| Model | HellaSwag | PIQA | WinoGrande | OpenBookQA | ARC-Easy | ARC-Challenge | BoolQ |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    first=source['checkpoints'][0]
    ts=source['protocol']['tasks']
    text+='| **L20-1B** | '+' | '.join(f"**{100*first['tasks'][t]['ours']:.2f}**" for t in ts)+' |\n'
    for job in SELECTED:
        row=lookup[job]
        text+='| '+label(job)+' | '+' | '.join(f"{100*row['tasks'][t]['baseline']:.2f}" for t in ts)+' |\n'
    text+='''
## Protocol and provenance

All numbers above are **our same-protocol re-evaluations**, not scores copied
from model cards: lm-eval 0.4.9, BF16, zero-shot, context 2,048, batch auto:4,
seeds 42/42/42/1234, no chat template. Training recipes and tokenizers differ.
Intervals cover benchmark-item sampling only, not training-seed variation, and
are not adjusted for multiple comparisons. Full raw model predictions are not
in the compact Git bundle; reproduction of aggregate evidence is not fresh inference.

- [Machine-readable presentation](comparison.json)
- [Original complete study and 252 task comparisons](../../pretraining/reports/metrics/efficiency-all-results-final-20260912.json)
- [Original report and claim boundaries](../../pretraining/reports/metrics/efficiency-all-results-final-20260912.md)
- [Language model card](../../pretraining/MODEL_CARD.md)
- [Reproduction instructions](../../docs/reproducibility.md)
- [Naming and presentation research](../../docs/presentation-research-2026-09-28.md)

Regenerate the summary with `python3 tools/build_language_showcase.py`.
Regenerate the figure with `python3 tools/plot_language_comparison.py` after
installing Matplotlib. `--check` validates the checked-in presentation and figure
hashes without Matplotlib or a GPU.
'''
    return text

def run(check: bool, root: Path = ROOT) -> dict:
    source=read_source(root);out=root/DEST
    outputs={'comparison.json':json.dumps(payload(source,root),indent=2)+'\n','README.md':report(source)}
    for name,content in outputs.items():
        p=out/name
        if check:
            if not p.is_file() or p.read_text()!=content:raise ValueError('Stale generated language presentation: '+name)
        else:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
    readme=root/'README.md';text=readme.read_text()
    if text.count(START)!=1 or text.count(END)!=1:raise ValueError('README language-table markers missing or duplicated')
    start=text.index(START)+len(START);end=text.index(END)
    if start>=end:raise ValueError('Invalid README marker order')
    expected='\n\n'+main_table(source)+'\n\n'
    if check:
        if text[start:end]!=expected:raise ValueError('README peer table differs from frozen evidence')
        fig=json.loads((root/'assets/language-comparison-manifest.json').read_text())
        if fig['source_sha256']!=digest(root/SOURCE) or fig['baseline_count']!=36:raise ValueError('Figure source mismatch')
        if set(fig['baseline_ids'])!={x['job'] for x in source['checkpoints']}:raise ValueError('Figure omitted a baseline')
        for name,h in fig['output_sha256'].items():
            if digest(root/'assets'/name)!=h:raise ValueError('Figure file changed')
        if fig['generator_sha256']!=digest(root/'tools/plot_language_comparison.py'):raise ValueError('Figure generator changed')
    else:readme.write_text(text[:start]+expected+text[end:])
    return {'status':'verified' if check else 'generated','baselines':36,'all_source_hashes_unchanged':True,'new_model_inference':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    print(json.dumps(run(args.check),indent=2))
