#!/usr/bin/env python3
"""Build complete human- and machine-readable bundles from verified summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            stream.write(text);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def inference(interval: list[float]) -> str:
    if interval[0] > 0:
        return 'ours_significant_win'
    if interval[1] < 0:
        return 'ours_significant_loss'
    return 'inconclusive'


def fmt(value: float, signed=False) -> str:
    return f'{value:+.4f}' if signed else f'{value:.4f}'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root',type=Path,required=True)
    parser.add_argument('--json-output',type=Path,required=True)
    parser.add_argument('--markdown-output',type=Path,required=True)
    args = parser.parse_args()

    root = args.archive_root.resolve()
    plan_path = root/'efficiency-evaluation-plan-20260912-v5.json'
    verification_path = root/'final-verification-20260912.json'
    plan = json.loads(plan_path.read_text())
    verification = json.loads(verification_path.read_text())
    if verification.get('status') != 'verified_complete' or verification.get('plan_sha256') != sha256(plan_path):
        raise ValueError('Final verification does not bind the selected plan')
    if len(plan.get('jobs',[])) != 36 or verification.get('complete_jobs') != 36:
        raise ValueError('Expected a complete 36-job bundle')

    ours_parameters = 1_100_048_384
    ours_tokens = 19_999_703_040
    rows = []
    counts = {'seven_task':{'ours_significant_win':0,'inconclusive':0,'ours_significant_loss':0},
              'six_task_without_boolq':{'ours_significant_win':0,'inconclusive':0,'ours_significant_loss':0}}
    for index,job in enumerate(plan['jobs'],1):
        summary_path = root/'results'/f'{job["id"]}-summary.json'
        summary = json.loads(summary_path.read_text())
        raw_path = root/'results'/f'{job["id"]}.json'
        binding = verification['result_bindings'][job['id']]
        if sha256(summary_path) != binding['summary_sha256'] or summary['result_sha256'] != binding['result_sha256']:
            raise ValueError(f'Verified summary binding changed: {job["id"]}')
        seven = summary['seven_task_macro']
        six = summary['six_task_without_boolq']
        seven_label = inference(seven['paired_95_ci_pp'])
        six_label = inference(six['paired_95_ci_pp'])
        counts['seven_task'][seven_label] += 1
        counts['six_task_without_boolq'][six_label] += 1
        rows.append({
            'index':index,'job':job['id'],'repo':job['repo'],'revision':job['revision'],
            'category':summary['category'],'training_tokens':summary['training_tokens'],
            'token_kind':summary['token_kind'],'total_parameters':summary['total_parameters'],
            'training_compute_proxy_ratio_vs_ours_6ND':summary['total_parameters']*summary['training_tokens']/(ours_parameters*ours_tokens),
            'seven_task_macro':{**seven,'inference':seven_label},
            'six_task_without_boolq':{**six,'inference':six_label},
            'tasks':summary['tasks'],'result_sha256':binding['result_sha256'],
            'summary_sha256':binding['summary_sha256'],'raw_result_remote_only':not raw_path.exists(),
            'limitations':summary['limitations'],
        })

    bundle = {
        'schema':1,'status':'verified_complete','created_unix':time.time(),
        'interpretation':'All differences are ours minus baseline in percentage points. A 95% CI wholly above/below zero is a significant win/loss; crossing zero is inconclusive.',
        'ours':{'parameters':ours_parameters,'training_tokens':ours_tokens,
                'seven_task_macro_percent':rows[0]['seven_task_macro']['ours']*100,
                'six_task_without_boolq_percent':rows[0]['six_task_without_boolq']['ours']*100},
        'protocol':plan['protocol'],'plan_sha256':verification['plan_sha256'],
        'final_verification_sha256':sha256(verification_path),'checkpoint_count':len(rows),
        'task_comparison_count':sum(len(row['tasks']) for row in rows),
        'inference_counts':counts,'checkpoints':rows,
        'evidence':{
            'queue_receipt_sha256':verification['queue_receipt_sha256'],
            'frontier_summary_sha256':verification['frontier_summary_sha256'],
            'frontier_svg_sha256':verification['frontier_svg_sha256'],
            'raw_results_remote_path':'/home/hhai/pretrain/evaluations/efficiency-20260912-v1/results',
            'local_summary_archive':str(root/'results'),
        },
        'limitations':verification['limitations'],
    }
    atomic_text(args.json_output,json.dumps(bundle,indent=2,ensure_ascii=False)+'\n')

    lines = [
        '# 1.1B pretraining efficiency: complete frozen-subset results','',
        'All 36 declared checkpoints completed. The fixed model is 1,100,048,384',
        'parameters trained on 19,999,703,040 tokens. Its seven-task macro is',
        f"{bundle['ours']['seven_task_macro_percent']:.4f}%; its six-task macro excluding BoolQ is {bundle['ours']['six_task_without_boolq_percent']:.4f}%.",'',
        'All differences are ours minus baseline in percentage points. `win` and',
        '`loss` mean the paired, task-stratified bootstrap 95% CI is wholly above',
        'or below zero; `tie` means the interval crosses zero. The 6ND ratio is an',
        'approximate training-compute proxy and excludes data, teacher, and search costs.','',
        '## Complete checkpoint-level table','',
        '| # | Checkpoint | Tokens (B) | Params (B) | 6ND ratio | Baseline 7-task | Ours−base 7-task, 95% CI | Result | Baseline 6-task | Ours−base 6-task, 95% CI | Result |',
        '|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---|',
    ]
    labels = {'ours_significant_win':'win','ours_significant_loss':'loss','inconclusive':'tie'}
    for row in rows:
        seven=row['seven_task_macro'];six=row['six_task_without_boolq']
        lines.append(
            f"| {row['index']} | `{row['job']}` | {row['training_tokens']/1e9:.5g} | {row['total_parameters']/1e9:.4f} | "
            f"{row['training_compute_proxy_ratio_vs_ours_6ND']:.4f}× | {seven['baseline']*100:.4f} | "
            f"{fmt(seven['ours_minus_baseline_pp'],True)} [{fmt(seven['paired_95_ci_pp'][0],True)}, {fmt(seven['paired_95_ci_pp'][1],True)}] | {labels[seven['inference']]} | "
            f"{six['baseline']*100:.4f} | {fmt(six['ours_minus_baseline_pp'],True)} [{fmt(six['paired_95_ci_pp'][0],True)}, {fmt(six['paired_95_ci_pp'][1],True)}] | {labels[six['inference']]} |"
        )
    lines += [
        '', '## Aggregate interpretation','',
        f"- Seven-task comparisons: {counts['seven_task']['ours_significant_win']} significant wins, {counts['seven_task']['inconclusive']} ties, {counts['seven_task']['ours_significant_loss']} significant losses.",
        f"- Six-task comparisons: {counts['six_task_without_boolq']['ours_significant_win']} significant wins, {counts['six_task_without_boolq']['inconclusive']} ties, {counts['six_task_without_boolq']['ours_significant_loss']} significant losses.",
        '- TinyLlama trajectory: ours significantly beats 1T, ties 1.5T and 2T, and significantly loses to 2.5T.',
        '- Near-compute results are mixed: ours beats TinyLlama 21B and DCLM baseline 14.4B, ties several DCLM/FineWeb points, and loses to QC7/FW3 at 18.0B and 21.6B.',
        '- WebOrganizer DCLM/domain-mix are strong higher-compute baselines; Phi-1.5 is a teacher-synthetic, much-higher-budget reference.',
        '', '## Evidence and boundaries','',
        f"- Plan SHA-256: `{verification['plan_sha256']}`",
        f"- Final verification SHA-256: `{sha256(verification_path)}`",
        f"- Queue receipt SHA-256: `{verification['queue_receipt_sha256']}`",
        f"- Frontier summary SHA-256: `{verification['frontier_summary_sha256']}`",
        '- The companion JSON contains all 252 task-level comparisons, sample win/loss/tie counts, task fingerprints, and result hashes.',
        '- The 36 raw per-example files remain on the remote host and are bound by SHA-256 in the final verification receipt.',
        '- These intervals cover benchmark-item sampling uncertainty, not training-seed variance, and are not adjusted for multiple comparisons.',
        '- BoolQ was not explicitly decontaminated from the original corpus, so the six-task sensitivity column is required.',
        '- This is a frozen declared subset, not a global census or proof of universal model superiority.','',
    ]
    atomic_text(args.markdown_output,'\n'.join(lines))
    print(json.dumps({'status':'complete','checkpoints':len(rows),'task_comparisons':bundle['task_comparison_count'],
                      'json_sha256':sha256(args.json_output),'markdown_sha256':sha256(args.markdown_output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
