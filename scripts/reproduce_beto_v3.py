"""Fixed V3 software reproduction on an already observed synthetic assessment.

No hyperparameter, epoch, calibration or review-policy selection is performed.
"""
from __future__ import annotations
import argparse,gc,json,time
from pathlib import Path
import numpy as np
from traveler_agent import beto_training_v3 as runner
from traveler_agent.beto_training_v3_model import EncoderV2,fit_head_v3
from traveler_agent.beto_classifier import BetoClassifier
from traveler_agent.beto_training_v2_model import load_model_v2
from traveler_agent.contextual_benchmark import train_baselines
from traveler_agent.corpus import CorpusError
from traveler_agent.labels import LABELS
from traveler_agent.research_metrics import score_matrix
from traveler_agent.synthetic_diversity import ignored_target,write_json

RAW_ATOL=.003  # Fixed in registration before the first V3 assessment.


def compare(key,rows,probabilities,reference):
    probabilities=np.asarray(probabilities,dtype=float)
    if (probabilities.shape!=(len(rows),3) or not np.isfinite(probabilities).all()
            or (probabilities<0).any() or not np.allclose(probabilities.sum(axis=1),1)):
        raise CorpusError('Incomplete or invalid reproduced probability matrix.')
    expected=[r for r in reference if r['model']==key]
    byid={r['review_id']:r for r in expected}
    if len(byid)!=len(expected) or set(byid)!={r['review_id'] for r in rows}:
        raise CorpusError('Reference membership changed or duplicated.')
    delta=0.
    for row,p in zip(rows,probabilities):
        old=byid[row['review_id']];q=np.asarray(old['probabilities'],dtype=float)
        if q.shape!=(3,) or not np.isfinite(q).all() or q.min()<0 or not np.isclose(q.sum(),1):
            raise CorpusError('Invalid fixed reference probabilities.')
        if (old['family']!=row['family'] or old['intended_label']!=row['intended_label']
                or old['prediction']!=LABELS[int(np.argmax(p))]):
            raise CorpusError('Fixed class, label or semantic membership drifted.')
        delta=max(delta,float(np.max(np.abs(p-q))))
    if delta>RAW_ATOL:raise CorpusError('Predeclared numerical reproduction tolerance exceeded.')
    return {'model':key,'rows':len(rows),'class_outcomes_exact':True,'max_raw_probability_difference':delta,'raw_atol_preassessment':RAW_ATOL}


def run(output,selection,full_selection,reference,*,download=False,primary_only=False):
    s=json.loads(selection.read_text(encoding='utf-8'));full=json.loads(full_selection.read_text(encoding='utf-8'))
    refs=json.loads(reference.read_text(encoding='utf-8'));commit=s['data_commit']
    if any(s[k]!=full[k] for k in ('records_sha256','selected_model_key','selected_adjusted','data_commit','protocol_commit')):
        raise CorpusError('Fixed selections disagree.')
    output=ignored_target(output,kinds=('artifacts',))
    if output.exists():raise CorpusError('New reproduction destination required.')
    output.mkdir(parents=True);registration=output/'registration';cache=output/'checkpoint'
    runner.register(registration,commit);parts,manifest,p=runner.verify(registration,commit)
    # Check membership before expensive replay. The reference remains observed.
    keys=[s['selected_model_key']] if primary_only else ['dummy','lr','svm','beto_frozen128','beto_frozen256',s['selected_model_key']]
    wanted={r['review_id'] for r in parts['assessment']}
    for key in keys:
        subset=[r for r in refs if r['model']==key]
        if len(subset)!=len(wanted) or {r['review_id'] for r in subset}!=wanted:raise CorpusError('Missing fixed model reference.')
    start=time.perf_counter();checks=[]
    replay=runner.replay(output/'selected-replay',registration,cache,commit,selection,42,download=download)
    model=load_model_v2(output/'selected-replay/model')
    checks.append(compare(s['selected_model_key'],parts['assessment'],score_matrix(model,[r['text'] for r in parts['assessment']]),refs))
    del model;gc.collect()
    if not primary_only:
        choices=full['classical_selection']
        models,_=train_baselines(parts['train'],parts['tune'],{'seed':42,'baselines':{
            'lr_C':[choices['lr']['selected']['C']],'svm_C':[choices['svm']['selected']['C']]}})
        for key,model in models.items():checks.append(compare(key,parts['assessment'],score_matrix(model,[r['text'] for r in parts['assessment']]),refs))
        del models,model;gc.collect()
        for length in (128,256):
            encoder=EncoderV2(cache,max_length=length,pooling='mean_l2',threads=2)
            x=runner.encode(encoder,parts['train']);epoch=full['head_selection'][str(length)]['selected']['epoch']
            weights,_=fit_head_v3(x,parts['train'],None,[],p,42,fixed_epoch=epoch)
            model=BetoClassifier(encoder,*weights,{})
            checks.append(compare('beto_frozen'+str(length),parts['assessment'],score_matrix(model,[r['text'] for r in parts['assessment']]),refs))
            del model,encoder,x;gc.collect()
    report={'scope':'REPRODUCTION_ONLY_OBSERVED_SYNTHETIC_ASSESSMENT_NOT_NEW_TEST','records_sha256':manifest['records_sha256'],
            'checks':checks,'rows_checked':sum(r['rows'] for r in checks),'primary_seed':42,
            'hyperparameters_or_epochs_reselected':False,'review_policy_selected':False,
            'wall_seconds':time.perf_counter()-start,'platform':replay['platform']}
    write_json(output/'reproduction.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selection',type=Path,default=Path('docs/research/beto-v3-selected-configuration.json'))
    p.add_argument('--full-selection',type=Path,default=Path('docs/research/beto-v3-selection.json'))
    p.add_argument('--reference',type=Path,default=Path('docs/research/beto-v3-predictions.json'))
    p.add_argument('--download',action='store_true');p.add_argument('--primary-only',action='store_true');args=p.parse_args()
    r=run(args.output,args.selection,args.full_selection,args.reference,download=args.download,primary_only=args.primary_only)
    print(json.dumps(r))
