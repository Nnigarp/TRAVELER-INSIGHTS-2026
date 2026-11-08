"""Research contracts and artificial fixtures; no human performance assertions."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from traveler_agent import beto_classifier as beto, contextual_data as data
from traveler_agent.contextual_benchmark import fit_frozen_head, selection_score
from traveler_agent.corpus import CorpusError
from traveler_agent.labels import LABELS


def test_new_experiment_has_disjoint_components_and_fixed_train_budget():
    parts,manifest,protocol=data.load_experiment()
    assert {k:len(v) for k,v in parts.items()}==protocol['role_rows']
    components={role:{r['semantic_component'] for r in rows} for role,rows in parts.items()}
    for left in components:
        for right in components:
            if left!=right:assert not components[left]&components[right]
    assert {r['source'] for r in parts['train']}=={'V3_TRAIN_ONLY_FIXED_BROAD_120'}
    assert len(components['train'])==20
    assert all(r['variant']<2 for r in parts['train'])
    assert manifest['duplicate_screen']['candidates']==[]
    assert len(manifest['duplicate_screen']['role_pair_screens'])==6
    assert manifest['human_gold'] is False
    assert all(r['intended_label'] is None for r in parts['ambiguity'])


@pytest.mark.parametrize('mutation',['human','component','duplicate'])
def test_invalid_provenance_and_cross_role_leakage_block_before_training(tmp_path,monkeypatch,mutation):
    catalog=json.loads(data.CATALOG.read_text())
    if mutation=='human':catalog['provenance']['human_authors']=True
    elif mutation=='component':
        catalog['families'][6]['semantic_component']=catalog['families'][0]['semantic_component']
    else:catalog['families'][6]['cases'][0]['text']=catalog['families'][0]['cases'][0]['text']
    path=tmp_path/'catalog.json';path.write_text(json.dumps(catalog))
    monkeypatch.setattr(data,'CATALOG',path)
    with pytest.raises(CorpusError):data.load_experiment()


def test_supervised_head_and_uncalibrated_probabilities_have_existing_contract():
    class Encoder:
        def encode(self,texts):return np.tile(np.eye(1,768,dtype=np.float32),(len(texts),1))
    model=beto.BetoClassifier(Encoder(),np.zeros((3,768)),[0,1,2],{})
    p=model.predict_proba(['Fixture artificial sin gold humano.'])
    assert p.shape==(1,3) and np.allclose(p.sum(axis=1),1)
    assert model.predict(['Fixture artificial.']).tolist()==['positivo']
    assert model.classes_.tolist()==list(LABELS)
    with pytest.raises(CorpusError):beto.BetoClassifier(Encoder(),np.zeros((2,768)),[0,0],{})
    with pytest.raises(CorpusError):model.predict_features(np.zeros((1,767)))


def test_head_really_optimizes_supervised_loss_and_selects_only_declared_epochs():
    pytest.importorskip('torch',reason='optional research environment; real training also checked by contextual CI')
    _,_,protocol=data.load_experiment()
    # Artificial separable vectors test optimization, not BETO representation.
    x=np.zeros((18,768),dtype=np.float32);y=np.arange(18)%3;x[np.arange(18),y]=1
    weights,record=fit_frozen_head(x,y,x,y,protocol)
    assert record['epochs'][-1]['train_cross_entropy']<record['epochs'][0]['train_cross_entropy']
    assert record['selected']['epoch'] in protocol['frozen_head']['selection_epochs']
    assert len(record['epochs'])==60
    model=beto.BetoClassifier(None,*weights,{})
    assert model.predict_features(x).shape==(18,3)
    assert record['encoder_trainable_parameters']==0


def test_selection_metrics_use_canonical_labels_and_do_not_mutate_input():
    p=np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]])
    original=p.copy();r=selection_score(np.arange(3),p)
    assert r['macro_f1']==1 and r['cross_entropy']>0
    assert np.array_equal(p,original)


@pytest.fixture
def artifact_fixture(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache=Path('artifacts/cache');cache.mkdir(parents=True)
    (cache/'fixture.txt').write_text('Artificial checkpoint fixture, never real BETO weights.')
    (cache/'verified_checkpoint.json').write_text('{}')
    (cache/'unverified-temporary.part').write_text('Must not be exported.')
    manifest={'model':beto.MODEL,'revision':beto.REVISION,'files':{'fixture.txt':{}}}
    monkeypatch.setattr(beto,'checkpoint',lambda *_args,**_kwargs:deepcopy(manifest))
    class Encoder:
        def __init__(self,*args,**kwargs):self.manifest=deepcopy(manifest)
        def encode(self,texts):return np.zeros((len(texts),768),dtype=np.float32)
    monkeypatch.setattr(beto,'BetoEncoder',Encoder)
    destination=Path('artifacts/version1')
    beto.export_classifier(destination,cache,np.zeros((3,768)),np.zeros(3),
        {'supervised':True,'mode':'fixture_only_not_real_training'})
    return destination


def test_portable_artifact_loads_non_executable_head_and_rejects_tamper(artifact_fixture):
    assert not (artifact_fixture/'checkpoint/unverified-temporary.part').exists()
    model=beto.load_classifier(artifact_fixture)
    assert np.allclose(model.predict_proba(['Fixture.']),1/3)
    with (artifact_fixture/'head.npz').open('ab') as f:f.write(b'altered')
    with pytest.raises(CorpusError,match='Cabeza modificada'):beto.load_classifier(artifact_fixture)


@pytest.mark.parametrize('mutation',['untrained','labels','revision','pooling'])
def test_untrained_or_incompatible_artifact_is_not_a_classifier(artifact_fixture,mutation):
    path=artifact_fixture/'model_manifest.json';metadata=json.loads(path.read_text())
    if mutation=='untrained':metadata['training']['supervised']=False
    elif mutation=='labels':metadata['classes']=['positivo','neutral','negativo']
    elif mutation=='revision':metadata['revision']='different'
    else:metadata['pooling']='unspecified'
    path.write_text(json.dumps(metadata))
    with pytest.raises(CorpusError,match='procedencia'):beto.load_classifier(artifact_fixture)


def test_saved_analysis_calls_no_model_and_detects_incomplete_predictions(tmp_path):
    from traveler_agent.contextual_analysis import analyze_saved
    parts,manifest,_=data.load_experiment()
    rows=[]
    for kind in ('lr','beto_frozen'):
        for row in parts['assessment']:
            p=np.eye(3)[LABELS.index(row['intended_label'])].tolist()
            rows.append({'model':kind,'review_id':row['review_id'],'family':row['family'],
                         'author_intent':row['intended_label'],'prediction':row['intended_label'],
                         'probabilities':p,'review':False})
    (tmp_path/'report.json').write_text(json.dumps({'data_manifest':manifest}))
    path=tmp_path/'assessment_predictions.json';path.write_text(json.dumps(rows))
    report=analyze_saved(tmp_path)
    assert report['models']['lr']['multiclass_brier_sum_per_row']==0
    assert report['models']['lr']['ece_10_equal_width_bins']==0
    assert report['models']['lr']['risk_policy_certified'] is False
    assert report['paired_family_bootstrap_macro_f1_difference_from_lr']['beto_frozen']['low_95']==0
    path.write_text(json.dumps(rows[:-1]))
    with pytest.raises(CorpusError,match='incompleta'):analyze_saved(tmp_path)


def test_reproduction_rejects_changed_class_and_probability_drift():
    from scripts.verify_contextual_reproduction import verify
    original=[{'model':'fixture','review_id':'x','family':'f','author_intent':'neutral',
               'prediction':'neutral','probabilities':[.1,.8,.1],'review':False}]
    assert verify(original,original)['maximum_probability_difference']==0
    changed=deepcopy(original);changed[0]['prediction']='positivo'
    with pytest.raises(ValueError,match='outcome'):verify(changed,original)
    changed=deepcopy(original);changed[0]['probabilities']=[.2,.7,.1]
    with pytest.raises(ValueError,match='drift'):verify(changed,original)
