"""Quality evidence, reviewer independence and lexical separation before training."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from traveler_agent import beto_training_v3_data as data
from traveler_agent.corpus import CorpusError,file_sha


def test_registered_pilot_and_full_plan_have_explicit_separate_budgets():
    p=data.protocol()
    assert sum(p['pilot_role_rows'].values())==564
    assert sum(p['target_5000_plan'][k] for k in p['pilot_role_rows'])==5000
    assert p['target_5000_plan']['status']=='PLANNED_NOT_GENERATED_NOT_TRAINED'
    assert p['human_validation'] is False


def test_near_copies_block_even_inside_a_related_family():
    rows=[{'text':'Llegamos al museo temprano y la sala lateral estaba disponible durante toda la visita.','review_id':'one','family':'same','role':'train'},
          {'text':'Llegamos al museo temprano y la sala lateral estaba disponible durante toda la visita!','review_id':'two','family':'same','role':'train'}]
    r=data.duplicate_audit(rows)
    assert len(r['candidates'])==1 and r['blocking_candidates']==1
    assert r['candidates'][0]['same_family'] is True


def test_nuisance_probe_cannot_fit_on_assessment_rows():
    with pytest.raises(CorpusError,match='TRAIN'):
        data.training_shortcuts([{'role':'assessment'}])


def quality_fixture(tmp_path,monkeypatch,shape=.4):
    p=data.protocol();manifest={'records_sha256':'records','protocol_sha256':'protocol'}
    checks={'near_duplicates':True,'shape_shortcuts':shape<=.45,'long_training_context':True,'repeated_endings':True}
    q={'manifest':manifest,'duplicates':{'candidates':[]},'shortcuts':{'length_and_punctuation':{'macro_f1':shape},'max_ending_family_fraction':.01},
       'tokenization':{'train':{'over_128_fraction':.33}},'checks':checks,'automatic_gates_pass':all(checks.values())}
    path=tmp_path/'quality.json';path.write_text(json.dumps(q));monkeypatch.setattr(data,'QUALITY',path)
    expected={'review_gates_pass':True,'checks':{'independent':True},'quarantined_families':[]}
    review={'automatic_audit_sha256':file_sha(path),'reviewers':[{'actual_task':'a','human':False,'synthetic_ai_review':True},
            {'actual_task':'b','human':False,'synthetic_ai_review':True}], 'adjudication_decisions':{},**expected}
    return p,manifest,q,review,expected


def test_admission_boolean_cannot_bypass_failed_style_gate(tmp_path,monkeypatch):
    p,m,q,r,_=quality_fixture(tmp_path,monkeypatch,shape=.499)
    r['admitted']=True
    with pytest.raises(CorpusError,match='automática'):
        data.validate_admission({},m,p,r)


def test_quality_hash_and_actual_reviewer_checks_are_required(tmp_path,monkeypatch):
    p,m,q,r,expected=quality_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(data,'review_statistics',lambda *_:expected)
    data.validate_admission({},m,p,r)
    r['automatic_audit_sha256']='forged'
    with pytest.raises(CorpusError,match='modificada'):
        data.validate_admission({},m,p,r)
    r['automatic_audit_sha256']=file_sha(data.QUALITY)
    expected['review_gates_pass']=False
    with pytest.raises(CorpusError,match='adjudicación'):
        data.validate_admission({},m,p,r)


def test_missing_quality_report_is_not_training_authorization(tmp_path,monkeypatch):
    p,m,_,r,_=quality_fixture(tmp_path,monkeypatch)
    data.QUALITY.unlink()
    with pytest.raises(CorpusError,match='incompleta'):
        data.validate_admission({},m,p,r)


def test_reviewer_identity_must_be_independent_before_any_annotation_use():
    reviewer={'actual_task':'same','annotations':[]}
    with pytest.raises(CorpusError,match='distintos'):
        data.review_statistics({}, {}, reviewer,deepcopy(reviewer),{})
