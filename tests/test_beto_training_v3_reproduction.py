"""Observed software-reference checks do not turn into model selection."""
from copy import deepcopy
import numpy as np
import pytest
from scripts.reproduce_beto_v3 import compare,RAW_ATOL
from traveler_agent.corpus import CorpusError


def sample():
    rows=[{'review_id':'one','family':'f','intended_label':'negativo'}]
    reference=[{'model':'fixed','review_id':'one','family':'f','intended_label':'negativo','prediction':'negativo','probabilities':[.7,.2,.1]}]
    return rows,reference


def test_fixed_reference_accepts_small_numerical_noise_without_changing_class():
    rows,refs=sample();r=compare('fixed',rows,np.array([[.701,.199,.1]]),refs)
    assert r['class_outcomes_exact'] and r['raw_atol_preassessment']==RAW_ATOL==.003


def test_fixed_reference_refuses_probability_drift_even_if_class_is_unchanged():
    rows,refs=sample()
    with pytest.raises(CorpusError,match='tolerance'):compare('fixed',rows,np.array([[.8,.1,.1]]),refs)


def test_fixed_reference_refuses_outcome_changes_and_duplicate_rows():
    rows,refs=sample()
    with pytest.raises(CorpusError,match='drifted'):compare('fixed',rows,np.array([[.1,.8,.1]]),refs)
    with pytest.raises(CorpusError,match='duplicated'):compare('fixed',rows,np.array([[.7,.2,.1]]),refs+deepcopy(refs))


def test_fixed_reference_refuses_changed_gold_intent_or_family():
    rows,refs=sample();refs[0]['intended_label']='neutral'
    with pytest.raises(CorpusError,match='drifted'):compare('fixed',rows,np.array([[.7,.2,.1]]),refs)


@pytest.mark.parametrize('probabilities',[
    np.empty((0,3)),np.array([[np.nan,.2,.1]]),np.array([[.9,.2,.1]]),np.array([[-.1,.7,.4]])])
def test_incomplete_or_invalid_replay_cannot_pass_reference_check(probabilities):
    rows,refs=sample()
    with pytest.raises(CorpusError,match='invalid reproduced'):compare('fixed',rows,probabilities,refs)


def test_fixed_classical_replay_never_reads_tune(monkeypatch):
    import scripts.reproduce_beto_v3 as script
    train=[{'role':'train'}]
    def fit(rows,score_rows,config):
        assert rows is score_rows is train
        assert config['baselines']=={'lr_C':[10.],'svm_C':[1.]}
        return {'fixed':'model'},{}
    monkeypatch.setattr(script,'train_baselines',fit)
    assert script.fit_fixed_baselines(train,{'lr':{'selected':{'C':10.}},'svm':{'selected':{'C':1.}}})=={'fixed':'model'}


def test_reference_requires_measured_provenance_with_matching_hash(tmp_path):
    import scripts.reproduce_beto_v3 as script
    from traveler_agent.corpus import file_sha
    path=tmp_path/'reference.json';path.write_text('[]')
    p={'predictions_sha256':file_sha(path),'records_sha256':'fixed','already_observed':True,'human_gold':False,
       'source_scope':'MEASURED_FIXED_REPRODUCTION_NOT_ORIGINAL_RAW_RECOVERY'}
    script.verify_reference(path,p,'fixed')
    path.write_text('[{}]')
    with pytest.raises(CorpusError,match='provenance'):script.verify_reference(path,p,'fixed')


@pytest.mark.parametrize('field,value',[('human_gold',True),('already_observed',False),('source_scope','OLD_AGGREGATES')])
def test_synthetic_reference_cannot_be_relabelled_as_new_or_human(tmp_path,field,value):
    import scripts.reproduce_beto_v3 as script
    from traveler_agent.corpus import file_sha
    path=tmp_path/'reference.json';path.write_text('[]')
    p={'predictions_sha256':file_sha(path),'records_sha256':'fixed','already_observed':True,'human_gold':False,
       'source_scope':'MEASURED_FIXED_REPRODUCTION_NOT_ORIGINAL_RAW_RECOVERY'}
    p[field]=value
    with pytest.raises(CorpusError,match='provenance'):script.verify_reference(path,p,'fixed')
