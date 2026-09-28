#!/usr/bin/env python3
"""Render every frozen language baseline; no inferred scores or hand-set colors."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from build_language_showcase import ROOT, SOURCE, read_source, digest

def main() -> None:
    data = read_source(ROOT)
    points = [[r['job'], r['training_compute_proxy_ratio_vs_ours_6ND'],
               r['seven_task_macro']['baseline']*100] for r in data['checkpoints']]
    assert len(points)==36 and len({p[0] for p in points})==36
    fig,ax=plt.subplots(figsize=(15.5,8.5))
    fig.subplots_adjust(left=.08,right=.97,top=.81,bottom=.25)
    ours=data['ours']['seven_task_macro_percent']
    ax.scatter([1.],[ours],s=370,marker='*',label='L20-1B · 20B tokens',zorder=6)
    groups=[('TinyLlama checkpoints',lambda n:n.startswith('tinyllama'),'o'),
            ('DataDecide recipes',lambda n:n.startswith('dd_'),'s'),
            ('WebOrganizer recipes',lambda n:n.startswith('weborganizer'),'^'),
            ('Pythia / OPT / Falcon',lambda n:n in {'pythia_1b','opt_1_3b','falcon_rw_1b'},'P'),
            ('Phi-1.5 · synthetic-data reference',lambda n:n=='phi_1_5','D')]
    plotted=[]
    for label,predicate,marker in groups:
        selected=sorted([p for p in points if predicate(p[0])],key=lambda p:p[1])
        plotted += [p[0] for p in selected]
        ax.plot([p[1] for p in selected],[p[2] for p in selected],marker=marker,
                markersize=6.5,linestyle='-' if label.startswith('TinyLlama') else 'None',
                linewidth=1.25,alpha=.8,label=label,zorder=3)
    assert len(plotted)==36 and len(set(plotted))==36
    lookup={p[0]:p for p in points}
    labels={
        'tinyllama_early_21b':('TinyLlama · 21B',(10,-15)),
        'tinyllama_early_105b':('105B',(0,-16)),
        'tinyllama_1t':('TinyLlama · 1T',(-12,-25)),
        'tinyllama_2_5t':('2.5T',(-10,12)),
        'pythia_1b':('Pythia-1B · 300B',(-8,-18)),
        'opt_1_3b':('OPT-1.3B · 180B',(-70,10)),
        'falcon_rw_1b':('Falcon-RW-1B · 350B',(10,7)),
        'phi_1_5':('Phi-1.5 · 150B',(10,-3)),
        'dd_dclm_baseline_qc_20p_42500':('DataDecide QC20 · 61B',(-28,17)),
    }
    for name,(text,offset) in labels.items():
        _,x,y=lookup[name]
        ax.annotate(text,(x,y),xytext=offset,textcoords='offset points',fontsize=9.5)
    ax.annotate(f'L20-1B\n{ours:.2f}% at 20B tokens',xy=(1.,ours),xytext=(.56,60.3),
                textcoords='data',fontsize=13,fontweight='bold',
                arrowprops={'arrowstyle':'-','linewidth':1.1})
    ax.set_xscale('log');ax.set_xlim(.42,170);ax.set_ylim(38.5,68)
    ax.xaxis.set_major_locator(FixedLocator([.5,1,2,5,10,20,50,100]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}×'))
    ax.minorticks_off();ax.set_yticks([40,45,50,55,60,65]);ax.grid(axis='y',alpha=.18)
    ax.tick_params(axis='both',length=0,pad=8,labelsize=10)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_ylabel('Seven-task zero-shot macro (%)  ↑',fontsize=11,labelpad=12)
    ax.set_xlabel('Approximate pretraining compute, 6ND / L20-1B  ·  logarithmic scale',fontsize=11,labelpad=13)
    ax.legend(loc='upper left',bbox_to_anchor=(-.007,-.20),ncol=3,frameon=False,
              fontsize=9.5,columnspacing=2.5,handlelength=2.3)
    fig.text(.08,.94,'A 1.1B language model. 20B tokens. One L20.',fontsize=23,fontweight='bold',ha='left')
    fig.text(.08,.889,'Same evaluation protocol · all 36 frozen baseline checkpoints shown · point estimates, not a global ranking',fontsize=11,ha='left')
    fig.text(.08,.045,'6ND is a compute proxy—not measured cost—and excludes data curation, teacher and experiment-search compute.\n'
             'Different training recipes/tokenizers; benchmark-item intervals and the six-task sensitivity are reported in the linked tables.',
             fontsize=9.5,linespacing=1.6,ha='left')
    out=ROOT/'assets';out.mkdir(exist_ok=True)
    png=out/'language-model-comparison.png';svg=out/'language-model-comparison.svg'
    fig.savefig(png,dpi=180);fig.savefig(svg,metadata={'Date':'2026-09-28'});plt.close(fig)
    record={'source':SOURCE,'source_sha256':digest(ROOT/SOURCE),'baseline_count':36,
            'baseline_ids':sorted(plotted),'total_points_including_ours':37,
            'source_score':'seven_task_macro.baseline','x_axis':'training_compute_proxy_ratio_vs_ours_6ND',
            'logarithmic_x_axis':True,'y_axis_is_point_estimate_not_confidence_interval':True,
            'new_inference':False,'matplotlib':matplotlib.__version__,
            'generator_sha256':digest(Path(__file__)),
            'output_sha256':{p.name:digest(p) for p in [png,svg]}}
    (out/'language-comparison-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
