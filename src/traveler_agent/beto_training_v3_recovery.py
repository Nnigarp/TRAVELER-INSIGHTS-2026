"""Rebuild archived V3 recipes after a workspace rollback, without reselection.

Only TRAIN is used for fitting. The old assessment is already observed. This
does not reconstruct missing original probabilities or create a new final test.
"""
from __future__ import annotations
import argparse,gc,json,time
from pathlib import Path

from . import beto_training_v3 as base
from .beto_training_v3_model import EncoderV2,fit_head_v3,train_encoder_v3,export_model,BudgetExceeded
from .contextual_benchmark import train_baselines,resource_report
from .corpus import CorpusError,file_sha
from .synthetic_diversity import ignored_target,write_json,canonical_sha

DATA_COMMIT='07935b6dc5eee6d62d2f62c6d67ce47b21cc58b9'
SELECTION_COMMIT='a6675d08fa9645725107f059443ddc0ed15d125f'
RECOVERY_PATH='src/traveler_agent/research_data/beto_v3_recovery_protocol.json'
EXPECTED_KEYS={'dummy','lr','svm','beto_frozen128','beto_frozen256',
               'beto_full256_budgeted','beto_reference_v1','beto_reference_v2'}


def fixed_recipes(selection):
    """Reject a different comparison or primary head/epoch recipe."""
    if (selection.get('data_commit')!=DATA_COMMIT or selection.get('protocol_commit')!=base.PROTOCOL_COMMIT
            or selection.get('selected_model_key')!='beto_full256_budgeted'
            or len(selection.get('model_entries',[]))!=len(EXPECTED_KEYS)
            or set(e['key'] for e in selection.get('model_entries',[]))!=EXPECTED_KEYS):
        raise CorpusError('Recovery requires the archived V3 comparison, not new candidates.')
    chosen=selection['selected_adjusted']
    if chosen['head_epoch']!=80 or chosen['selected_epoch']!=1 or chosen['configuration']['mode']!='full':
        raise CorpusError('Do not change the archived supervised recipe.')
    return {e['key']:e for e in selection['model_entries']}


def fit_fixed_classics(train,selection):
    if not train or any(r.get('role')!='train' for r in train):
        raise CorpusError('Fixed recovery fits TRAIN only.')
    choices=selection['classical_selection']
    # The helper scores its only candidate; use TRAIN for that discarded score.
    # Neither C nor any other decision depends on tune in this reconstruction.
    models,_=train_baselines(train,train,{'seed':42,'baselines':{
        'lr_C':[choices['lr']['selected']['C']], 'svm_C':[choices['svm']['selected']['C']]}})
    return models


def verify_protocol(commit,manifest,selection):
    registered=base._git_json(commit,RECOVERY_PATH)
    local=json.loads(Path(RECOVERY_PATH).read_text(encoding='utf-8'))
    if (registered!=local or registered.get('records_sha256')!=manifest['records_sha256']
            or registered.get('original_selection_commit')!=SELECTION_COMMIT
            or registered.get('assessment_already_observed') is not True
            or registered.get('new_hyperparameter_or_epoch_selection') is not False
            or registered.get('max_preparation_seconds')!=900
            or registered.get('max_assessment_and_replica_seconds')!=900):
        raise CorpusError('Recovery protocol changed or not registered before reconstruction.')
    if (selection['records_sha256']!=manifest['records_sha256']
            or registered.get('fixed_selection_decisions_sha256')!=canonical_sha(base._selection_decisions(selection))):
        raise CorpusError('Do not substitute another corpus during recovery.')
    fixed_recipes(selection)
    return registered


def reserve_observed(manifest):
    marker=base._observation_marker(manifest['records_sha256'])
    marker.parent.mkdir(parents=True,exist_ok=True)
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')).get('records_sha256')!=manifest['records_sha256']:
            raise CorpusError('Inconsistent existing observation marker.')
    else:
        write_json(marker,{'records_sha256':manifest['records_sha256'],
            'scope':'PRIOR_OBSERVED_ASSESSMENT_ORIGINAL_RAW_ARTIFACTS_UNAVAILABLE',
            'evidence':'docs/research/beto-v3-prior-reported-aggregates.json',
            'new_test':False,'original_probabilities_reconstructed':False})
    return marker


def prepare(out,cache,recovery_commit):
    parts,manifest,p=base.data.load_data(require_review=True)
    original=base._git_json(SELECTION_COMMIT,base.SELECTION_PATH)
    verify_protocol(recovery_commit,manifest,original);recipes=fixed_recipes(original)
    if json.loads(Path(base.SELECTION_PATH).read_text(encoding='utf-8'))!=original:
        raise CorpusError('Preserve the original versioned tune-only selection unchanged.')
    selected_file=Path('docs/research/beto-v3-selected-configuration.json')
    selected=json.loads(selected_file.read_text(encoding='utf-8'))
    if any(selected[k]!=original[k] for k in selected):raise CorpusError('Archived selections disagree.')
    out=ignored_target(out,kinds=('artifacts',));cache=ignored_target(cache,kinds=('artifacts',))
    if out.exists():raise CorpusError('Use a new reconstruction directory, never overwrite an attempt.')
    out.mkdir(parents=True);start=time.perf_counter();deadline=start+900
    base.register(out/'registration',DATA_COMMIT);reserve_observed(manifest)
    write_json(out/'manifest.json',manifest);write_json(out/'protocol.json',p)
    entries={};features={}
    def progress(stage,**values):
        event={'stage':stage,'elapsed_seconds':time.perf_counter()-start,**values}
        base._snapshot(out/'recovery-progress.json',event);print(json.dumps(event),flush=True)
    def bind(key,path=None,pipeline=None,training=None):
        e=recipes[key];entries[key]=base._entry(out,key,e['kind'],artifact=path,pipeline=pipeline,
            configuration=e.get('configuration'),training=training or {'scope':'FIXED_TRAIN_ONLY_RECOVERY'})
    try:
        progress('FIXED_CLASSICS_TRAIN_ONLY')
        models=base._bounded_call(deadline,'fixed_classics',fit_fixed_classics,parts['train'],original)
        for key,model in models.items():bind(key,pipeline=model)
        del models,model;gc.collect()
        for length in (128,256):
            key='beto_frozen'+str(length);progress('FIXED_FROZEN',model=key)
            encoder=base._bounded_call(deadline,'frozen_load',EncoderV2,cache,max_length=length,pooling='mean_l2',threads=2)
            x=base.encode(encoder,parts['train'],deadline);features[length]=x
            epoch=original['head_selection'][str(length)]['selected']['epoch']
            weights,log=fit_head_v3(x,parts['train'],None,[],p,42,fixed_epoch=epoch,deadline=deadline)
            path=out/'models'/key;config=recipes[key]['configuration']
            base._bounded_call(deadline,'frozen_export',export_model,path,cache,weights,config,
                base._training_metadata(manifest,DATA_COMMIT,42,'OBSERVED_FIXED_RECOVERY',epoch))
            bind(key,path,training=log);del encoder;gc.collect()
        progress('FIXED_PRIMARY_SUPERVISED_REPLAY')
        log=base.replay(out/'primary-replay',out/'registration',cache,DATA_COMMIT,selected_file,42,deadline=deadline)
        bind(original['selected_model_key'],out/'primary-replay/model',training=log)
        progress('FIXED_REFERENCE_V1')
        rp=json.loads(json.dumps(p));rp['head']['dropout']=0.
        weights,log=fit_head_v3(features[128],parts['train'],None,[],rp,42,fixed_epoch=30,deadline=deadline)
        key='beto_reference_v1';path=out/'models'/key
        base._bounded_call(deadline,'reference_v1_export',export_model,path,cache,weights,recipes[key]['configuration'],
            base._training_metadata(manifest,DATA_COMMIT,42,'OBSERVED_FIXED_V1_RECOVERY',30))
        bind(key,path,training=log)
        progress('FIXED_REFERENCE_V2')
        key='beto_reference_v2';config=recipes[key]['configuration']
        weights,head=fit_head_v3(features[128],parts['train'],None,[],p,42,fixed_epoch=80,deadline=deadline)
        encoder=base._bounded_call(deadline,'reference_v2_load',EncoderV2,cache,max_length=128,pooling='mean_l2',threads=2)
        state=out/'reference-v2.safetensors'
        weights,log=train_encoder_v3(encoder,parts['train'],[],weights,config,p,42,state,fixed_epoch=1,deadline=deadline)
        path=out/'models'/key
        base._bounded_call(deadline,'reference_v2_export',export_model,path,cache,weights,config,
            base._training_metadata(manifest,DATA_COMMIT,42,'OBSERVED_FIXED_V2_RECOVERY',80,1),state=state)
        bind(key,path,training={'head':head,'encoder':log});del encoder;gc.collect()
        # Original metrics/decisions/cost remain archived evidence from a6675d0.
        # Only artifact bindings change; no old raw prediction is fabricated.
        rebound=json.loads(json.dumps(original));rebound['model_entries']=[entries[e['key']] for e in original['model_entries']]
        rebound['recovery']={'scope':'FIXED_RECIPES_ON_ALREADY_OBSERVED_CORPUS_NOT_NEW_TEST',
            'protocol_registration_commit':recovery_commit,'original_selection_commit':SELECTION_COMMIT,
            'original_selection_source_sha256':file_sha(Path(base.SELECTION_PATH)),
            'original_raw_results_available':False,'archived_development_cost_is_not_current_rebuild_cost':True,
            'new_hyperparameter_or_epoch_selection':False,'tune_used_for_fitting_or_selection':False,
            'cost':{'wall_seconds':time.perf_counter()-start,'wall_seconds_cap':900,**resource_report()}}
        base.verify_selection_commit(rebound,SELECTION_COMMIT,reproduce_observed=True)
        base._bounded_call(deadline,'recovery_bindings_write',write_json,out/'development-selection.json',rebound)
        write_json(out/'selected-configuration.json',selected)
        progress('RECONSTRUCTION_COMPLETED_NO_ASSESSMENT_PREDICTIONS')
        return rebound
    except BaseException as exc:
        write_json(out/'recovery-preparation-interruption.json',{'status':'INTERRUPTED_COMPUTE_BUDGET' if isinstance(exc,BudgetExceeded) else 'ERROR',
            'error_type':type(exc).__name__,'reason':str(exc),'assessment_predictions_performed':False,
            'scope':'FIXED_OBSERVED_RECONSTRUCTION_NOT_MODEL_SELECTION','wall_seconds':time.perf_counter()-start})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--recovery-commit',required=True)
    args=parser.parse_args();r=prepare(args.output,args.cache,args.recovery_commit)
    print(json.dumps({'prepared':r['selected_model_key'],'scope':r['recovery']['scope'],'cost':r['recovery']['cost']}))
