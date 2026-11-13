from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from traveler_agent.model import _text_features
from traveler_agent.labels import LABELS
from traveler_agent.beto_training_v3_representation import (BoundTrainTfidf,fit_represented_baselines,
    training_fingerprint,validate_state)
from traveler_agent.corpus import CorpusError
from scripts.export_beto_v3_representation import snapshot


def fixture():
    rows=[{'review_id':str(i),'role':'train','family':'f'+str(i//3),'semantic_component':'f'+str(i//3),
           'intended_label':LABELS[i%3],'text':f'Opinión {i} hotel '+('sucio','horario','agradable')[i%3]} for i in range(9)]
    def state(fit,held=None):return snapshot(SimpleNamespace(named_steps={'features':_text_features().fit([r['text'] for r in fit])}),fit,held)
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(rows,[r['intended_label'] for r in rows],groups=[r['family'] for r in rows]))
    choices={k:{'selected':{'C':1.}} for k in ('lr','svm')}
    s={'schema':'beto-v3-fitted-train-representation-v1','training_input_sha256':training_fingerprint(rows),
       'source_roles':['train'],'human_gold':False,'classical_C':{'lr':1.,'svm':1.},'full':state(rows),
       'svm_folds':[state([rows[i] for i in fit],[rows[i] for i in held]) for fit,held in folds]}
    return rows,choices,s


def test_replay_does_not_redo_frequency_cutoff_or_read_other_roles(monkeypatch):
    rows,choices,s=fixture()
    def forbidden(*a,**kw):pytest.fail('Fitted representation replay must not select frequency-cutoff ties again.')
    monkeypatch.setattr(CountVectorizer,'_limit_features',forbidden)
    models=fit_represented_baselines(rows,choices,s)
    for model in models.values():
        p=model.predict_proba([r['text'] for r in rows]);assert p.shape==(9,3) and np.allclose(p.sum(axis=1),1)
    rows[0]['role']='assessment'
    with pytest.raises(CorpusError,match='TRAIN'):fit_represented_baselines(rows,choices,s)


def test_global_vocabulary_cannot_be_used_for_a_held_fold():
    rows,_,s=fixture();vec=BoundTrainTfidf([s['full']],'words')
    with pytest.raises(CorpusError,match='Unregistered'):vec.fit_transform([r['text'] for r in rows[:6]])


def test_vocabulary_cannot_include_a_token_absent_from_its_fit_fold():
    rows,_,s=fixture();s=deepcopy(s);s['full']['features']['words'].append('held_only_token')
    with pytest.raises(CorpusError,match='absent'):BoundTrainTfidf([s['full']],'words').fit_transform([r['text'] for r in rows])


def test_fold_state_preserves_family_separation_and_exact_membership():
    rows,choices,s=fixture();validate_state(rows,s,choices)
    s['svm_folds'][0]['fit_review_ids']=s['full']['fit_review_ids']
    with pytest.raises(CorpusError,match='membership'):validate_state(rows,s,choices)


def test_changed_training_text_cannot_reuse_another_representation():
    rows,choices,s=fixture();rows[0]['text']='otro contexto'
    with pytest.raises(CorpusError,match='registered TRAIN'):validate_state(rows,s,choices)
