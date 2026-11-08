"""Post-hoc descriptive analysis of SAVED predictions, with no model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contextual_data import load_experiment
from .corpus import CorpusError, file_sha
from .labels import LABELS
from .research_metrics import family_confusions, bootstrap_indices, bootstrap_macro, interval
from .synthetic_diversity import ignored_target, write_json


def analyze_saved(root: Path):
    report=json.loads((root/'report.json').read_text(encoding='utf-8'))
    predictions=json.loads((root/'assessment_predictions.json').read_text(encoding='utf-8'))
    parts,manifest,_=load_experiment()
    if report['data_manifest']['records_sha256']!=manifest['records_sha256']:
        raise CorpusError('Predicciones de otro catálogo.')
    rows=parts['assessment'];truth=np.asarray([r['intended_label'] for r in rows])
    by_id={r['review_id']:r for r in rows};groups={}
    for record in predictions:
        rid=record['review_id'];model=record['model']
        if (rid not in by_id or record['author_intent']!=by_id[rid]['intended_label'] or
                record['family']!=by_id[rid]['family'] or rid in groups.setdefault(model,{})):
            raise CorpusError('Predicciones sin correspondencia única con el catálogo.')
        p=np.asarray(record['probabilities'])
        if (p.shape!=(3,) or not np.isfinite(p).all() or (p<0).any() or
                not np.isclose(p.sum(),1) or record['prediction']!=LABELS[p.argmax()]):
            raise CorpusError('Probabilidades/predicción inconsistentes.')
        groups[model][rid]=record
    statistics,samples={},{}
    for model,entries in groups.items():
        if set(entries)!=set(by_id):raise CorpusError('Evaluación incompleta.')
        saved=[entries[r['review_id']] for r in rows]
        p=np.asarray([r['probabilities'] for r in saved]);ids=np.asarray([LABELS.index(t) for t in truth])
        one_hot=np.eye(3)[ids];conf=p.max(axis=1);correct=p.argmax(axis=1)==ids
        bins=[];ece=0.
        for i in range(10):
            selected=(conf>=i/10)&((conf<(i+1)/10) if i<9 else (conf<=1))
            n=int(selected.sum())
            if n:
                acc=float(correct[selected].mean());mean=float(conf[selected].mean())
                ece+=n/len(p)*abs(acc-mean)
                bins.append({'lower':i/10,'upper':(i+1)/10,'n':n,'accuracy':acc,'mean_confidence':mean})
        names,cubes=family_confusions(rows,truth,[r['prediction'] for r in saved])
        draws=bootstrap_indices(len(names),2000,42)
        samples[model]=bootstrap_macro(cubes,draws)
        accepted=[r for r in saved if not r['review']]
        errors=sum(r['author_intent']!=r['prediction'] for r in accepted)
        statistics[model]={'negative_log_likelihood':float(-np.log(np.clip(p[np.arange(len(p)),ids],1e-12,1)).mean()),
            'multiclass_brier_sum_per_row':float(np.square(p-one_hot).sum(axis=1).mean()),
            'ece_10_equal_width_bins':ece,'reliability_bins':bins,
            'family_bootstrap_macro_f1':interval(samples[model]),
            'accepted':len(accepted),'accepted_errors':errors,
            'zero_error_iid_upper_95_reference_only':1-.05**(1/len(accepted)) if accepted and errors==0 else None,
            'risk_policy_certified':False}
    differences={model:interval(value-samples['lr']) for model,value in samples.items() if model!='lr'}
    return {'scope':'POSTHOC_DESCRIPTIVE_SAVED_SYNTHETIC_PREDICTIONS_NOT_MODEL_OR_POLICY_SELECTION',
        'assessment_is_now_observed':True,'source_report_sha256':file_sha(root/'report.json'),
        'source_predictions_sha256':file_sha(root/'assessment_predictions.json'),'models':statistics,
        'paired_family_bootstrap_macro_f1_difference_from_lr':differences,
        'limits':['Ten synthetic families, one training seed; bootstrap is not human population inference.',
                  'ECE binning and bootstrap are descriptive analyses chosen after assessment, never selection criteria.',
                  'The zero-error binomial reference assumes independent fixed-policy Bernoulli decisions; related rows and policy selection invalidate a risk certification.',
                  'No fitted calibrator, changed threshold or abstention policy is admitted.']}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    output=ignored_target(args.out,kinds=('artifacts',))
    write_json(output,analyze_saved(args.run))
    print(json.dumps({'status':'SAVED_PREDICTIONS_ANALYZED_NO_MODEL_CALLS','output':str(args.out)}))


if __name__=='__main__':main()
