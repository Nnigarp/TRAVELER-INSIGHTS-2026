from copy import deepcopy
import numpy as np
import pytest
from traveler_agent import beto_training_v3_continuation as c
from traveler_agent.corpus import CorpusError


def fixture():
    members=[{'review_id':'r'+str(i),'family':'family','intended_label':label} for i,label in enumerate(c.base.LABELS)]
    saved=[]
    for i,row in enumerate(members):
        p=[.1,.1,.1];p[i]=.8
        saved.append({**row,'model':'lr','prediction':row['intended_label'],'probabilities':p,'calibrated_probabilities':p})
    truth=[r['intended_label'] for r in members]
    partial={'complete':False,'predictions':saved,'results':{'lr':{'classification':c.base.summarize_predictions(truth,truth)}}}
    return members,[{'key':'lr'},{'key':'svm'}],partial,{'completed_model_keys':['lr']}


def test_completed_probability_prefix_is_preserved_without_fitting():
    args=fixture();before=deepcopy(args)
    assert c.verify_prefix(*args)==['lr'] and args==before


@pytest.mark.parametrize('mutation',['probability','label','membership','nonprefix','metric'])
def test_a_changed_original_result_cannot_be_hidden_as_continuation(mutation):
    members,entries,partial,interruption=fixture()
    if mutation=='probability':partial['predictions'][0]['probabilities']=[.1,.8,.1]
    elif mutation=='label':partial['predictions'][0]['intended_label']='positivo'
    elif mutation=='membership':partial['predictions'][0]['review_id']='other'
    elif mutation=='nonprefix':entries.reverse()
    else:partial['results']['lr']['classification']['macro_f1']=.7
    with pytest.raises(CorpusError):c.verify_prefix(members,entries,partial,interruption)


def test_missing_inference_never_calls_training_or_policy_selection(monkeypatch):
    def forbidden(*a,**kw):pytest.fail('A fixed inference continuation must never refit or select.')
    monkeypatch.setattr(c.base,'fit_temperature',forbidden);monkeypatch.setattr(c.base,'select_policy',forbidden)
    monkeypatch.setattr(c.base,'replay',forbidden)
    rows=[{'review_id':str(i),'family':'f','semantic_component':'f','intended_label':label,'text':str(i),
           'region':'fixture','length_band':'short','phenomena':[]} for i,label in enumerate(c.base.LABELS)]
    class Model:
        classes_=np.asarray(c.base.LABELS)
        def predict_proba(self,texts):return np.asarray([[.8 if i==int(t) else .1 for i in range(3)] for t in texts])
    result,saved=c.evaluate_one(Model(),'fixture',{'assessment':rows,'ambiguity_assessment':[{'text':'0'}],'tune':rows},
                              {'temperature':{'temperature':1.},'policy':{'selected':None}}, {'policy':{}},None)
    assert result['classification']['macro_f1']==1. and len(saved)==3
