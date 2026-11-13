"""Extract original fitted TRAIN feature state from this run's hashed artifacts."""
import argparse,json
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from traveler_agent import beto_training_v3 as base
from traveler_agent.beto_training_v3_representation import STATE_PATH,training_fingerprint,validate_state
from traveler_agent.corpus import file_sha
from traveler_agent.synthetic_diversity import canonical_sha


def snapshot(model,fit,held=None):
    features=model.named_steps['features']
    return {'fit_review_ids':[r['review_id'] for r in fit],
        'held_review_ids':[r['review_id'] for r in held] if held is not None else [],
        'fit_documents_sha256':canonical_sha([r['text'] for r in fit]),
        'features':{name:vectorizer.get_feature_names_out().tolist() for name,vectorizer in features.transformer_list}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    parts,manifest,_=base.data.load_data(require_review=True);train=parts['train']
    selection=base._git_json('a6675d08fa9645725107f059443ddc0ed15d125f',base.SELECTION_PATH)
    entries=json.loads((a.run/'evaluation-models.json').read_text(encoding='utf-8'));bykey={e['key']:e for e in entries}
    lr=base._load_entry(a.run,bykey['lr']);svm=base._load_entry(a.run,bykey['svm'])
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(
        [r['text'] for r in train],[r['intended_label'] for r in train],groups=[r['semantic_component'] for r in train]))
    state={'schema':'beto-v3-fitted-train-representation-v1','records_sha256':manifest['records_sha256'],
        'training_input_sha256':training_fingerprint(train),'source_roles':['train'],'human_gold':False,
        'original_selection_commit':'a6675d08fa9645725107f059443ddc0ed15d125f',
        'classical_C':{k:selection['classical_selection'][k]['selected']['C'] for k in ('lr','svm')},
        'source_artifact_sha256':{k:bykey[k]['sha256'] for k in ('lr','svm')},
        'full':snapshot(lr,train),
        'svm_folds':[snapshot(est.estimator,[train[i] for i in fit],[train[i] for i in held])
                     for est,(fit,held) in zip(svm.calibrated_classifiers_,folds)],
        'reason':'Preserve the already fitted TRAIN frequency-cutoff tie choices across CPU dispatch; no new feature selection or assessment access.'}
    validate_state(train,state,selection['classical_selection'])
    with STATE_PATH.open('x',encoding='utf-8') as f:json.dump(state,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({'path':str(STATE_PATH),'sha256':file_sha(STATE_PATH),'vocabulary_sizes':
        [{k:len(v) for k,v in s['features'].items()} for s in [state['full'],*state['svm_folds']]]}))
