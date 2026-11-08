"""Plot only saved synthetic assessment metrics; never fit or choose a model."""
import argparse
import json
from pathlib import Path


def plot(report_path,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    report=json.loads(report_path.read_text(encoding='utf-8'))['results']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1,1.1]})
    x=np.arange(3);width=.24
    colors=['#64748b','#0f766e','#4f46e5']
    for j,(kind,label,color) in enumerate(zip(['lr','svm','beto_frozen_mean_l2_128'],['TF-IDF + LR','TF-IDF + SVM','BETO fijo + cabeza'],colors)):
        values=[report[f'{kind}_{arm}']['classification']['macro_f1'] for arm in ['existing120','new120','mixed240']]
        bars=axes[0].bar(x+(j-1)*width,values,width,label=label,color=color)
        axes[0].bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
    axes[0].set(xticks=x,xticklabels=['Históricos 120','Nuevos 120','Mezcla 240'],ylim=(0,1.04),ylabel='Macro-F1',title='Cantidad y composición del entrenamiento')
    axes[0].legend(loc='upper center',ncol=3,frameon=False,fontsize=8)
    keys=['lr_mixed240','svm_mixed240','beto_frozen_mean_l2_128_mixed240','beto_top2_low128','beto_selected_seed7','beto_selected_seed101']
    values=[report[key]['classification']['negative_to_positive_count'] for key in keys]
    bars=axes[1].bar(range(len(keys)),values,color=[*colors,'#312e81','#818cf8','#a5b4fc'])
    axes[1].bar_label(bars,padding=4)
    axes[1].set(xticks=range(len(keys)),xticklabels=['LR\n240','SVM\n240','BETO fijo\n240','2 bloques\nsem. 42','2 bloques\nsem. 7','2 bloques\nsem. 101'],ylim=(0,max(10,max(values)+1)),ylabel='Errores sobre 30 negativos',title='Negativos clasificados como positivos')
    for ax in axes:ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18)
    fig.suptitle('BETO Training V2 · 90 textos sintéticos nuevos',fontsize=15,fontweight='bold',y=.98)
    fig.text(.06,.03,'Etiquetas de intención del asistente, sin validación humana. Semilla 42 fijada antes de evaluar; no se elige la mejor semilla.',fontsize=9,color='#475569')
    fig.tight_layout(rect=(0,.09,1,.93));out.parent.mkdir(parents=True,exist_ok=True)
    for extension in ('png','svg'):fig.savefig(out.with_suffix('.'+extension),dpi=180,facecolor='white')
    svg=out.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();plot(args.report,args.out)
