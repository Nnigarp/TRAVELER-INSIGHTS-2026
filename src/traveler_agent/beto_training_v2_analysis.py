"""Descriptive paired-family and seed analysis on saved V2 predictions only."""
import argparse
import json
from pathlib import Path

import numpy as np

from .beto_training_v2_data import load_data
from .corpus import CorpusError,file_sha
from .labels import LABELS
from .research_metrics import family_confusions,bootstrap_indices,bootstrap_macro,interval
from .synthetic_diversity import write_json,ignored_target


def analyze(root):
    report=json.loads((root/'assessment.json').read_text());saved=json.loads((root/'assessment-predictions.json').read_text())
    parts,manifest,_=load_data()
    if report['data_manifest']['records_sha256']!=manifest['records_sha256']:raise CorpusError('Resultados de otro catálogo.')
    rows=parts['assessment'];ids={r['review_id'] for r in rows};truth=[r['intended_label'] for r in rows];models={}
    for record in saved:
        key=record['model'];rid=record['review_id']
        if rid in models.setdefault(key,{}):raise CorpusError('Predicción duplicada.')
        models[key][rid]=record
    paired={};samples={}
    for key,predictions in models.items():
        if set(predictions)!=ids:raise CorpusError('Predicciones incompletas.')
        ordered=[predictions[r['review_id']] for r in rows]
        if any(r['intended_label']!=y for r,y in zip(ordered,truth)):raise CorpusError('Intenciones desalineadas.')
        names,cubes=family_confusions(rows,truth,[r['prediction'] for r in ordered])
        samples[key]=bootstrap_macro(cubes,bootstrap_indices(len(names),2000,42))
        paired[key]={'macro_f1_family_bootstrap':interval(samples[key]),'families':len(names)}
    reference=samples['lr_mixed240']
    for key in paired:paired[key]['difference_from_lr_mixed240']=interval(samples[key]-reference)
    primary=report['selected_model_key'];seed_keys=[primary,'beto_selected_seed7','beto_selected_seed101']
    observed=[k for k in seed_keys if k in report['results']]
    scores=[report['results'][k]['classification']['macro_f1'] for k in observed]
    return {'scope':'SAVED_SYNTHETIC_PREDICTIONS_ONLY_NO_FITTING_OR_POLICY_SELECTION',
        'source_report_sha256':file_sha(root/'assessment.json'),'paired_family_bootstrap':paired,
        'selected_configuration_seed_summary':{'model_keys':observed,'macro_f1':scores,'mean_macro_f1':float(np.mean(scores)),
          'sample_std_macro_f1':float(np.std(scores,ddof=1)) if len(scores)>1 else None,
          'negative_to_positive':[report['results'][k]['classification']['negative_to_positive_count'] for k in observed],
          'seed_selected_for_UI':42,'best_seed_selection_performed':False},
        'limits':['Synthetic author intent and residual writing-style correlations; no human-population interval.',
                  'Three seeds do not establish broad optimizer stability; all share the same synthetic evaluation.',
                  'Source and diversity are confounded in existing120 versus new120 comparisons.',
                  'This assessment is now observed, not a reusable fresh final test.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();write_json(ignored_target(args.out,kinds=('artifacts',)),analyze(args.run))


if __name__=='__main__':main()
