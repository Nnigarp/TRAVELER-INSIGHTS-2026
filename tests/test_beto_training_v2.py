"""Role isolation, safe artifacts and actual head optimization on synthetic fixtures."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from traveler_agent import beto_training_v2_data as data,beto_training_v2_policy as policy
from traveler_agent.beto_training_v2_model import fit_head,train_encoder,_allowed_keys
from traveler_agent.corpus import CorpusError
from traveler_agent.labels import LABELS


def test_registered_data_roles_components_and_new_train_budget():
    parts,m,p=data.load_data()
    assert {k:len(v) for k,v in parts.items()}==p['role_rows']
    assert m['counts']['train']['classes']=={label:80 for label in LABELS}
    assert m['counts']['train']['components']==60
    assert m['duplicates']['candidate_count']==0 and len(m['duplicates']['pairs'])==21
    seen={}
    for role,rows in parts.items():
        for row in rows:
            assert row['semantic_component'] not in seen or seen[row['semantic_component']]==role
            seen[row['semantic_component']]=role
    assert all(r['intended_label'] is None for r in parts['ambiguity_assessment'])
    assert m['human_gold'] is False


def test_registration_detects_catalog_bytes_changed_before_any_fitting(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    root=Path('artifacts/registration');data.register(root)
    data.verify_registration(root)
    manifest=json.loads((root/'data-manifest.json').read_text());manifest['records_sha256']='0'*64
    (root/'data-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(CorpusError,match='cambiados'):data.verify_registration(root)


@pytest.mark.parametrize('role',['assessment','calibration','policy_dev'])
def test_final_or_calibration_roles_cannot_select_head_or_encoder(role):
    wrong=[{'role':role}]
    with pytest.raises(CorpusError):fit_head(None,[{'role':'train'}],None,wrong,{},42)
    with pytest.raises(CorpusError):train_encoder(None,[{'role':'train'}],wrong,None,{}, {},42,None)


def test_only_declared_encoder_parameters_are_trainable():
    fake=SimpleNamespace(state_dict=lambda:{'embeddings.word_embeddings.weight':None,'encoder.layer.0.weight':None,
                                           'encoder.layer.10.weight':None,'encoder.layer.11.weight':None})
    assert _allowed_keys(fake,'frozen')==set()
    assert _allowed_keys(fake,'top2')=={'encoder.layer.10.weight','encoder.layer.11.weight'}
    assert len(_allowed_keys(fake,'full'))==4
    with pytest.raises(CorpusError):_allowed_keys(fake,'unknown')


def test_positive_text_guard_uses_observed_text_not_sentiment_gold():
    rows=[{'text':'No fue una mala visita.'},{'text':'Qué experiencia tan agradable.'}]
    p=np.array([[.01,.01,.98],[.01,.01,.98]])
    spec={'global':.6,'positive':.95,'margin':.1,'guard':True}
    assert policy.reviewed(rows,p,spec).tolist()==[True,False]
    spec['guard']=False;assert not policy.reviewed(rows,p,spec).any()


def test_policy_refuses_assessment_selection_and_never_calls_review_all_a_solution():
    _,_,p=data.load_data()
    row={'role':'assessment','text':'Fixture.','intended_label':'positivo','semantic_component':'fixture'}
    amb={'role':'ambiguity_policy','text':'Fixture ambiguo.'}
    with pytest.raises(CorpusError):policy.select([row],[[.1,.1,.8]],[amb],[[.3,.4,.3]],p)
    row['role']='policy_dev'
    result=policy.select([row],[[.1,.1,.8]],[amb],[[.3,.4,.3]],p)
    assert result['selected'] is None and result['status'].startswith('NO_POLICY_ADMITTED')


def test_stage_head_updates_and_fixed_epoch_replay_preserves_selected_weights():
    pytest.importorskip('torch',reason='executed in dedicated CPU/Windows research jobs')
    _,_,protocol=data.load_data();protocol=deepcopy(protocol)
    protocol['head'].update(epochs=4,selection_epochs=[2,4])
    rows=[{'role':'train','intended_label':LABELS[i%3]} for i in range(18)]
    tune=[{**r,'role':'tune'} for r in rows]
    x=np.zeros((18,768),dtype=np.float32);x[np.arange(18),np.arange(18)%3]=1
    weights,log=fit_head(x,rows,x,tune,protocol,42)
    replay,_=fit_head(x,rows,x,tune,protocol,42,fixed_epoch=log['selected']['epoch'])
    assert np.array_equal(weights[0],replay[0]) and np.array_equal(weights[1],replay[1])
    assert log['history'][-1]['train_cross_entropy']<log['history'][0]['train_cross_entropy']


def test_v2_runtime_dispatch_keeps_v1_loader_contract(tmp_path,monkeypatch):
    from traveler_agent import model_runtime
    import traveler_agent.beto_training_v2_model as models
    path=tmp_path/'v2';path.mkdir();(path/'model_manifest.json').write_text(json.dumps({'schema':models.SCHEMA}))
    sentinel=object();monkeypatch.setattr(models,'load_model_v2',lambda p:sentinel)
    assert model_runtime._load(str(path),'fixture-new-version') is sentinel
    model_runtime._load.cache_clear()


def test_staged_encoder_updates_only_allowed_blocks_and_replays_original_schedule(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch');pytest.importorskip('transformers')
    import traveler_agent.beto_training_v2_model as models
    from safetensors.torch import load_file
    protocol=json.loads(data.PROTOCOL.read_text())
    config={'mode':'top2','epochs':2,'encoder_lr':.001,'head_lr':.001,'weight_decay':.01,'head_dropout':.1,'label_smoothing':0.0}
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embeddings=torch.nn.Linear(3,3,bias=False)
            self.encoder=torch.nn.Module();self.encoder.layer=torch.nn.ModuleList([torch.nn.Linear(3,3,bias=False) for _ in range(12)])
            for parameter in self.parameters():torch.nn.init.eye_(parameter);parameter.requires_grad_(False)
    class FixtureEncoder:
        def __init__(self):self.model=Tiny()
        def tokens(self,texts):return torch.eye(3)[[int(t) for t in texts]]
        def pooled(self,x):
            x=self.model.embeddings(x)
            for layer in self.model.encoder.layer:x=x+.1*layer(x)
            return torch.nn.functional.pad(torch.nn.functional.normalize(x,dim=1),(0,765))
        def encode(self,texts):
            self.model.eval()
            with torch.no_grad():return self.pooled(self.tokens(texts)).numpy()
    rows=[{'role':'train','text':str(i%3),'intended_label':LABELS[i%3]} for i in range(12)]
    tune=[{**r,'role':'tune'} for r in rows[:3]]
    weight=np.zeros((3,768),np.float32);weight[:,:3]=np.eye(3);bias=np.zeros(3,np.float32)
    # Fixed equal tune scores force epoch1, testing replay with a 2-epoch LR plan.
    monkeypatch.setattr(models,'tuning_score',lambda *_:{'macro_f1':.5,'cross_entropy':1.,'negative_to_positive':1})
    encoder=FixtureEncoder();before={k:v.clone() for k,v in encoder.model.state_dict().items()}
    _,log=train_encoder(encoder,rows,tune,(weight,bias),config,protocol,42,tmp_path/'primary.safetensors')
    assert log['selected']['epoch']==1 and log['probe_tensor_l2_update']>0 and log['first_encoder_gradient_norm']>0
    allowed=_allowed_keys(encoder.model,'top2')
    assert all(torch.equal(before[k],v) for k,v in encoder.model.state_dict().items() if k not in allowed)
    replay=FixtureEncoder()
    train_encoder(replay,rows,tune,(weight,bias),config,protocol,42,tmp_path/'replay.safetensors',fixed_epoch=1)
    first,second=load_file(str(tmp_path/'primary.safetensors')),load_file(str(tmp_path/'replay.safetensors'))
    assert all(torch.equal(first[k],second[k]) for k in first)


def test_untrained_v2_manifest_is_rejected_before_loading_encoder(tmp_path,monkeypatch):
    import traveler_agent.beto_training_v2_model as models
    monkeypatch.chdir(tmp_path)
    root=Path('artifacts/untrained');root.mkdir(parents=True)
    (root/'model_manifest.json').write_text(json.dumps({'schema':models.SCHEMA,'training':{'supervised':False}}))
    monkeypatch.setattr(models,'EncoderV2',lambda *_args,**_kwargs:pytest.fail('Untrained artifact must not load weights'))
    with pytest.raises(CorpusError,match='no supervisado'):models.load_model_v2(root)
