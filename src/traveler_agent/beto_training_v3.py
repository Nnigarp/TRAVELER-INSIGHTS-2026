"""Registered V3 pilot: admit, develop, lock, calibrate, assess and fixed replay."""
from __future__ import annotations
import argparse,gc,json,math,platform,re,subprocess,time
from pathlib import Path
import numpy as np
from . import beto_training_v3_data as data
from .beto_training_v3_model import (EncoderV2,fit_head_v3,train_encoder_v3,export_model,BudgetExceeded)
from .beto_training_v2_model import score_rank,tuning_score
from .beto_training_v2 import _entry,_load_entry
from .beto_training_v2_policy import select as select_policy,describe as describe_policy
from .contextual_benchmark import train_baselines,resource_report
from .corpus import CorpusError,file_sha
from .labels import LABELS
from .research_metrics import score_matrix,summarize_predictions,slice_metrics,review_mask
from .research_v3_selective import fit_temperature,temperature_transform,probability_metrics
from .synthetic_diversity import ignored_target,write_json,canonical_sha

PROTOCOL_COMMIT='e4c52c18c146355ab955debc1255e680af8df2a4'
SELECTION_PATH='docs/research/beto-v3-selection.json'
ASSESSMENT_PATH='docs/research/beto-v3-assessment.json'
FRESH_SCOPE='FRESH_SYNTHETIC_PILOT_ASSESSMENT'
REPRODUCTION_SCOPE='REPRODUCTION_OF_OBSERVED_SYNTHETIC_PILOT_NOT_A_FRESH_TEST'


def _check_deadline(deadline,stage):
    if deadline is not None:
        if not math.isfinite(deadline):raise CorpusError('Plazo absoluto finito requerido.')
        if time.perf_counter()>=deadline:raise BudgetExceeded(stage)


def _bounded_call(absolute_deadline,stage,operation,*args,**kwargs):
    """A blocking call may finish late; never start another stage after it."""
    _check_deadline(absolute_deadline,stage+':before')
    result=operation(*args,**kwargs)
    _check_deadline(absolute_deadline,stage+':after')
    return result


def _inference_call(deadline,stage,operation,*args,evidence=None,assessment=False,**kwargs):
    _check_deadline(deadline,stage+':before')
    if evidence is not None:
        evidence['last_inference_stage']=stage
        if assessment:evidence['assessment_prediction_calls_started']+=1
    result=operation(*args,**kwargs)
    if evidence is not None and assessment:evidence['assessment_prediction_calls_returned']+=1
    _check_deadline(deadline,stage+':after')
    return result


def _score_with_deadline(model,texts,deadline,stage,*,evidence=None,assessment=False):
    # Bounded batches also protect classical backends and permit a stop between
    # BETO calls, without relying on any backend's internal batching behavior.
    scores=[]
    for start in range(0,len(texts),8):
        scores.append(_inference_call(deadline,f'{stage}:batch_{start}',score_matrix,
            model,texts[start:start+8],evidence=evidence,assessment=assessment))
    _check_deadline(deadline,stage+':complete')
    return np.concatenate(scores) if scores else np.empty((0,len(LABELS)))


def _snapshot(path,value):
    """Mutable progress/selection snapshots; observation markers stay exclusive."""
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def _git_json(commit,path):
    if not re.fullmatch('[0-9a-f]{40}',commit or ''):raise CorpusError('Commit completo de selección requerido.')
    try:
        content=subprocess.check_output(['git','show',f'{commit}:{path}'],stderr=subprocess.DEVNULL)
        value=json.loads(content)
    except (subprocess.CalledProcessError,json.JSONDecodeError,UnicodeDecodeError) as exc:
        raise CorpusError('Selección no disponible en el commit Git registrado.') from exc
    if not isinstance(value,dict):raise CorpusError('Selección Git inválida.')
    return value


def _selection_decisions(selection):
    """Only registered decisions, excluding metrics, paths, times and hashes."""
    def candidate(row):
        if row is None:return None
        return {key:row.get(key) for key in ('id','configuration','head_epoch','selected_epoch')}
    return {'records_sha256':selection['records_sha256'],'data_commit':selection['data_commit'],
        'protocol_commit':selection['protocol_commit'],'selected_model_key':selection['selected_model_key'],
        'selected_adjusted':candidate(selection.get('selected_adjusted')),
        'best_overall':candidate(selection.get('best_overall')),
        'classical_C':{key:value['selected']['C'] for key,value in selection['classical_selection'].items()},
        'head_epochs':{str(key):value['selected']['epoch'] for key,value in selection['head_selection'].items()},
        'candidates':[{**candidate(row),'status':row['status']} for row in selection['candidates']],
        'comparison_models':sorted([{'key':row['key'],'kind':row['kind'],'configuration':row.get('configuration')}
                                    for row in selection['model_entries']],key=lambda row:row['key'])}


def verify_selection_commit(selection,commit,*,reproduce_observed=False):
    registered=_git_json(commit,SELECTION_PATH)
    try:
        matches=(_selection_decisions(selection)==_selection_decisions(registered)
                 if reproduce_observed else selection==registered)
    except (KeyError,TypeError) as exc:raise CorpusError('Decisiones de selección incompletas.') from exc
    if not matches:raise CorpusError('La selección difiere del commit registrado; no predecir assessment.')
    return registered


def _observation_marker(records_sha):
    if not re.fullmatch('[0-9a-f]{64}',records_sha or ''):raise CorpusError('Hash de corpus inválido.')
    return Path.cwd()/'artifacts/beto-v3-observed'/f'{records_sha}.json'


def _report_records(report):
    return report.get('records_sha256',report.get('data_manifest',{}).get('records_sha256'))


def known_observations(records_sha):
    """Changing --output cannot turn an already observed corpus into a test."""
    evidence=[];marker=_observation_marker(records_sha)
    if marker.exists():evidence.append({'source':str(marker),'records_sha256':records_sha})
    # Read the tracked working report and HEAD: deleting/editing a working copy
    # cannot conceal the committed observation. Untracked reports are not proof.
    tracked=subprocess.run(['git','ls-files','--error-unmatch',ASSESSMENT_PATH],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
    if tracked.returncode==0:
        path=Path(ASSESSMENT_PATH)
        if path.exists():
            try:report=json.loads(path.read_text(encoding='utf-8'))
            except (json.JSONDecodeError,UnicodeDecodeError) as exc:raise CorpusError('Informe assessment registrado ilegible.') from exc
            if _report_records(report)==records_sha:evidence.append({'source':ASSESSMENT_PATH,'records_sha256':records_sha})
        committed=subprocess.run(['git','show',f'HEAD:{ASSESSMENT_PATH}'],capture_output=True,check=False)
        if committed.returncode==0:
            try:report=json.loads(committed.stdout)
            except (json.JSONDecodeError,UnicodeDecodeError) as exc:raise CorpusError('Informe assessment Git ilegible.') from exc
            if _report_records(report)==records_sha:evidence.append({'source':'HEAD:'+ASSESSMENT_PATH,'records_sha256':records_sha})
    return evidence


def _require_unobserved(records_sha,*,reproduce_observed=False):
    observed=known_observations(records_sha)
    if observed and not reproduce_observed:
        raise CorpusError('Corpus assessment ya observado/iniciado; solo --reproduce-observed con decisiones registradas.')
    return observed


def register(out,commit):
    parts,manifest,p=data.load_data(require_review=True)
    data.verify_git_registration(PROTOCOL_COMMIT);data.verify_git_registration(commit,manifest)
    out=ignored_target(out,kinds=('artifacts',))
    if out.exists():raise CorpusError('Registro existente; conservarlo.')
    out.mkdir(parents=True);write_json(out/'manifest.json',manifest);write_json(out/'protocol.json',p)
    write_json(out/'registration.json',{'protocol_commit':PROTOCOL_COMMIT,'data_commit':commit,
               'records_sha256':manifest['records_sha256'],'synthetic':True,'human_gold':False})
    return manifest


def verify(registration,commit):
    parts,manifest,p=data.load_data(require_review=True)
    data.verify_git_registration(PROTOCOL_COMMIT);data.verify_git_registration(commit,manifest)
    if (json.loads((registration/'manifest.json').read_text(encoding='utf-8'))!=manifest
            or json.loads((registration/'protocol.json').read_text(encoding='utf-8'))!=p):
        raise CorpusError('Datos o protocolo difieren del registro; no entrenar.')
    return parts,manifest,p


def encode(encoder,rows,deadline=None):
    result=[]
    for start in range(0,len(rows),8):
        result.append(_inference_call(deadline,f'feature_encoding:batch_{start}',encoder.encode,
                                      [r['text'] for r in rows[start:start+8]]))
    return np.concatenate(result)


def _training_metadata(manifest,commit,seed,mode,head_epoch,encoder_epoch=0):
    return {'supervised':True,'mode':mode,'experiment':'BETO_V3_QUALITY_PILOT','seed':seed,
            'head_epoch':head_epoch,'encoder_epoch':encoder_epoch,'records_sha256':manifest['records_sha256'],
            'data_registration_commit':commit,'protocol_registration_commit':PROTOCOL_COMMIT,
            'label_origin':'AI_AUTHOR_INTENT_WITH_BLIND_AI_REVIEW_NOT_HUMAN_GOLD'}


def train_classical(train,tune,p,deadline=None):
    if any(r['role']!='train' for r in train) or any(r['role']!='tune' for r in tune):
        raise CorpusError('Clásicos: solo TRAIN/tune separados.')
    selected={};records={}
    for c in p['baseline_C']:
        models,_=_bounded_call(deadline,'classical_train',train_baselines,train,tune,
                              {'seed':p['primary_seed'],'baselines':{'lr_C':[c],'svm_C':[c]}})
        for kind,model in models.items():
            if kind=='dummy' and kind in selected:continue
            score={'C':None if kind=='dummy' else c,**tuning_score(tune,
                _score_with_deadline(model,[r['text'] for r in tune],deadline,'classical_tune:'+kind))}
            record=records.setdefault(kind,{'candidates':[],'selected':None});record['candidates'].append(score)
            if record['selected'] is None or score_rank(score)>score_rank(record['selected']):
                selected[kind]=model;record['selected']=score
    return selected,records


def develop(out,registration,cache,commit,*,download=False,reproduce_observed=False):
    parts,manifest,p=verify(registration,commit);out=ignored_target(out,kinds=('artifacts',))
    _require_unobserved(manifest['records_sha256'],reproduce_observed=reproduce_observed)
    if reproduce_observed:
        raise CorpusError('No reseleccionar desarrollo observado; usar replay de la selección fija registrada.')
    if out.exists():raise CorpusError('Desarrollo existente; no sobrescribir.')
    out.mkdir(parents=True);start=time.perf_counter();deadline=start+p['compute_budget']['max_development_wall_seconds']
    def progress(stage,**kwargs):
        event={'stage':stage,'elapsed_seconds':time.perf_counter()-start,**kwargs}
        _snapshot(out/'progress.json',event);print(json.dumps(event),flush=True)
    write_json(out/'manifest.json',manifest);write_json(out/'protocol.json',p)
    progress('CLASSICAL_TRAIN_TUNE_ONLY')
    models,classic=train_classical(parts['train'],parts['tune'],p,deadline)
    entries=[_entry(out,key,'classical',pipeline=model,training={'source':'V3_ADMITTED_TRAIN','rows':len(parts['train'])}) for key,model in models.items()]
    del models;gc.collect();head_weights={};head_logs={};features={};candidates=[];selected_weights={};epoch_seconds=None
    for length in (128,256):
        progress('FROZEN_FEATURES',max_length=length)
        encoder=_bounded_call(deadline,'frozen_encoder_load',EncoderV2,cache,download=download,
                              max_length=length,pooling='mean_l2',threads=2)
        train_features=encode(encoder,parts['train'],deadline);dev_features=encode(encoder,parts['tune'],deadline)
        weights,log=fit_head_v3(train_features,parts['train'],dev_features,parts['tune'],p,p['primary_seed'],deadline=deadline)
        head_weights[length]=weights;head_logs[length]=log;features[length]=train_features
        config=next(c for c in p['candidates'] if c['id']==f'frozen{length}')
        path=out/'models'/config['id'];export_model(path,cache,weights,config,
            _training_metadata(manifest,commit,42,'v3_frozen_head',log['selected']['epoch']))
        entry=_entry(out,'beto_'+config['id'],'beto',artifact=path,configuration=config,training={'head':log})
        entries.append(entry);candidates.append({'id':config['id'],'configuration':config,'status':'EXECUTED',
            'head_epoch':log['selected']['epoch'],'selected_epoch':0,'tune':log['selected']['tune'],'training':log})
        del encoder,train_features,dev_features;gc.collect()
    for config in p['candidates']:
        if config['mode']=='frozen':continue
        progress('ADJUSTED_CANDIDATE',candidate=config['id'])
        remaining=deadline-time.perf_counter()
        if remaining<=0 or (config['mode']=='full' and (epoch_seconds is None or remaining<epoch_seconds*3+30)):
            candidates.append({'id':config['id'],'configuration':config,'status':'NOT_EXECUTED_COMPUTE_BUDGET',
                               'remaining_seconds':remaining,'projected_full_epoch_seconds':None if epoch_seconds is None else epoch_seconds*3})
            continue
        encoder=EncoderV2(cache,max_length=config['max_length'],pooling=config['pooling'],threads=2)
        state=out/'states'/f'{config["id"]}.safetensors'
        try:
            weights,log=train_encoder_v3(encoder,parts['train'],parts['tune'],head_weights[256],config,p,42,state,deadline=deadline)
            if epoch_seconds is None:epoch_seconds=log['history'][0]['elapsed_seconds']
            selected_weights[config['id']]=weights
            candidates.append({'id':config['id'],'configuration':config,'status':'EXECUTED','head_epoch':head_logs[256]['selected']['epoch'],
                               'selected_epoch':log['selected']['epoch'],'tune':log['selected']['tune'],'training':log})
        except BudgetExceeded as exc:
            candidates.append({'id':config['id'],'configuration':config,'status':'INTERRUPTED_COMPUTE_BUDGET','stage':exc.stage})
        finally:del encoder;gc.collect()
    executed=[r for r in candidates if r['status']=='EXECUTED']
    adjusted=[r for r in executed if r['configuration']['mode']!='frozen']
    chosen=max(adjusted,key=lambda r:score_rank(r['tune'])) if adjusted else None
    overall=max(executed,key=lambda r:score_rank(r['tune']))
    if chosen:
        path=out/'models'/chosen['id'];export_model(path,cache,selected_weights[chosen['id']],chosen['configuration'],
            _training_metadata(manifest,commit,42,'v3_staged_'+chosen['configuration']['mode'],chosen['head_epoch'],chosen['selected_epoch']),
            state=out/'states'/f'{chosen["id"]}.safetensors')
        entries.append(_entry(out,'beto_'+chosen['id'],'beto',artifact=path,configuration=chosen['configuration'],training=chosen))
    # Persist the valid primary selection BEFORE any optional historical recipe.
    # References never determine the selection and cannot erase it on failure.
    references=[{'key':key,'status':'NOT_EXECUTED_COMPUTE_BUDGET'}
                for key in ('beto_reference_v1','beto_reference_v2')]
    selection={'schema':'beto-v3-pilot-selection','data_commit':commit,'protocol_commit':PROTOCOL_COMMIT,
        'records_sha256':manifest['records_sha256'],'classical_selection':classic,'head_selection':head_logs,
        'candidates':candidates,'selected_adjusted':chosen,'best_overall':overall,
        'selected_model_key':None if chosen is None else 'beto_'+chosen['id'],'model_entries':entries,
        'historical_recipe_references':references,'assessment_observed':False,
        'cost':{'wall_seconds':time.perf_counter()-start,**resource_report()},
        'environment':{'python':platform.python_version(),'platform':platform.platform()}}
    _require_unobserved(manifest['records_sha256'])
    write_json(out/'development-selection.json',selection)
    write_json(out/'selected-configuration.json',{k:selection[k] for k in
        ('schema','data_commit','protocol_commit','records_sha256','selected_adjusted','selected_model_key')})
    def persist_selection():
        selection['cost']={'wall_seconds':time.perf_counter()-start,**resource_report()}
        _snapshot(out/'development-selection.json',selection)
    def failed_reference(record,exc):
        record.update(status='INTERRUPTED_COMPUTE_BUDGET' if isinstance(exc,BudgetExceeded) else 'FAILED_OPTIONAL_REFERENCE',
                      error_type=type(exc).__name__,reason=str(exc))
        if isinstance(exc,BudgetExceeded):record['stage']=exc.stage
    # Fixed historical recipes are retrained on this same NEW TRAIN, not scored
    # on or selected using historical evaluations. They are not old binaries.
    if deadline-time.perf_counter()>30:
        try:
            rp=json.loads(json.dumps(p));rp['head']['dropout']=0.0
            weights,log=_bounded_call(deadline,'reference_v1_head',fit_head_v3,
                features[128],parts['train'],None,[],rp,42,fixed_epoch=30,deadline=deadline)
            config={'id':'reference_v1_frozen128','mode':'frozen','pooling':'mean_l2','max_length':128}
            path=out/'models'/config['id'];_bounded_call(deadline,'reference_v1_export',export_model,
                path,cache,weights,config,_training_metadata(manifest,commit,42,'FIXED_V1_RECIPE_RETRAINED_ON_NEW_TRAIN',30))
            entries.append(_entry(out,'beto_reference_v1','beto',artifact=path,configuration=config,training=log))
            references[0]['status']='EXECUTED'
        except Exception as exc:failed_reference(references[0],exc)
        finally:persist_selection()
    if chosen and epoch_seconds and deadline-time.perf_counter()>epoch_seconds+30:
        encoder=None
        try:
            config={'id':'reference_v2_top2_128','mode':'top2','pooling':'mean_l2','max_length':128,'epochs':4,'encoder_lr':1e-5,
                    'head_lr':5e-4,'weight_decay':.01,'head_dropout':.1,'label_smoothing':0.,'hard_weight':1.,'patience':1}
            weights,hlog=_bounded_call(deadline,'reference_v2_head',fit_head_v3,
                features[128],parts['train'],None,[],p,42,fixed_epoch=80,deadline=deadline)
            encoder=_bounded_call(deadline,'reference_v2_encoder_load',EncoderV2,
                cache,max_length=128,pooling='mean_l2',threads=2)
            state=out/'states/reference-v2.safetensors'
            weights,log=_bounded_call(deadline,'reference_v2_encoder',train_encoder_v3,
                encoder,parts['train'],[],weights,config,p,42,state,fixed_epoch=1,deadline=deadline)
            path=out/'models'/config['id'];_bounded_call(deadline,'reference_v2_export',export_model,
                path,cache,weights,config,_training_metadata(manifest,commit,42,'FIXED_V2_RECIPE_RETRAINED_ON_NEW_TRAIN',80,1),state=state)
            entries.append(_entry(out,'beto_reference_v2','beto',artifact=path,configuration=config,training={'head':hlog,'encoder':log}))
            references[1]['status']='EXECUTED'
        except Exception as exc:failed_reference(references[1],exc)
        finally:del encoder;gc.collect();persist_selection()
    persist_selection()
    progress('DEVELOPMENT_COMPLETED',selected=selection['selected_model_key'])
    return selection


def replay(out,registration,cache,commit,selection_file,seed,*,download=False,deadline=None):
    start=time.perf_counter()
    parts,manifest,p=verify(registration,commit);s=json.loads(selection_file.read_text(encoding='utf-8'))
    if s['records_sha256']!=manifest['records_sha256'] or s['data_commit']!=commit:raise CorpusError('Selección de otro corpus.')
    if s.get('protocol_commit')!=PROTOCOL_COMMIT:raise CorpusError('Selección de otro protocolo.')
    chosen=s.get('selected_adjusted')
    if not chosen:raise CorpusError('No hay candidato ajustado ejecutado; no inventar réplica.')
    _check_deadline(deadline,'replay_start')
    cap=min(900,p['compute_budget']['max_replica_and_assessment_wall_seconds'])
    deadline=min(deadline,start+cap) if deadline is not None else start+cap
    _check_deadline(deadline,'replay_start')
    observed=known_observations(manifest['records_sha256'])
    scope=REPRODUCTION_SCOPE if observed else 'FIXED_TRAIN_ONLY_REPLAY_NOT_AN_ASSESSMENT'
    out=ignored_target(out,kinds=('artifacts',))
    if out.exists():raise CorpusError('Réplica existente; conservarla.')
    out.mkdir(parents=True);config=chosen['configuration']
    encoder=None
    try:
        encoder=_bounded_call(deadline,'replay_encoder_load',EncoderV2,cache,download=download,
                              max_length=config['max_length'],pooling=config['pooling'],threads=2)
        x=encode(encoder,parts['train'],deadline)
        weights,hlog=_bounded_call(deadline,'replay_head',fit_head_v3,x,parts['train'],None,[],p,
                                  seed,fixed_epoch=chosen['head_epoch'],deadline=deadline)
        state=out/'state.safetensors'
        weights,log=_bounded_call(deadline,'replay_encoder',train_encoder_v3,encoder,parts['train'],[],
            weights,config,p,seed,state,fixed_epoch=chosen['selected_epoch'],deadline=deadline)
        _bounded_call(deadline,'replay_export',export_model,out/'model',cache,weights,config,
            _training_metadata(manifest,commit,seed,'V3_FIXED_REPLAY',chosen['head_epoch'],chosen['selected_epoch']),state=state)
        result={'seed':seed,'fixed_configuration':chosen,'head':hlog,'encoder':log,'assessment_used':False,
                'selection_retuned':False,'artifact':str(out/'model'),'memory':resource_report(),'platform':platform.platform(),
                'scope':scope,'cost':{'wall_seconds':time.perf_counter()-start,'wall_seconds_cap':cap,
                                     'effective_wall_seconds_cap':deadline-start}}
        _bounded_call(deadline,'replay_report_write',write_json,out/'replay.json',result)
        return result
    except BudgetExceeded as exc:
        if (out/'replay.json').exists():(out/'replay.json').rename(out/'replay-interrupted-report.json')
        write_json(out/'replay-interruption.json',{'status':'INTERRUPTED_COMPUTE_BUDGET','stage':exc.stage,
            'seed':seed,'scope':scope,'assessment_used':False,'selection_retuned':False,
            'artifact_complete_within_budget':False,'wall_seconds':time.perf_counter()-start,'wall_seconds_cap':cap,
            'effective_wall_seconds_cap':deadline-start})
        raise
    finally:del encoder;gc.collect()


def acquire_assessment_lock(out,metadata,*,reproduce_observed=False):
    records_sha=metadata['records_sha256']
    _require_unobserved(records_sha,reproduce_observed=reproduce_observed)
    global_marker=_observation_marker(records_sha);global_marker.parent.mkdir(parents=True,exist_ok=True)
    try:
        with global_marker.open('x',encoding='utf-8') as f:
            json.dump({**metadata,'output':str(out.resolve())},f,ensure_ascii=False,indent=2)
    except FileExistsError as exc:
        if not reproduce_observed:
            raise CorpusError('Corpus assessment ya iniciado/observado en otro destino.') from exc
    marker=out/'assessment-started.json'
    try:
        with marker.open('x',encoding='utf-8') as f:json.dump(metadata,f,ensure_ascii=False,indent=2)
    except FileExistsError as exc:raise CorpusError('Assessment ya iniciado/observado; no sobrescribir o reintentar silenciosamente.') from exc
    return marker


def assess(out,registration,cache,commit,*,replicas=True,selection_commit=None,reproduce_observed=False):
    start=time.perf_counter()
    parts,manifest,p=verify(registration,commit);out=ignored_target(out,kinds=('artifacts',))
    s=json.loads((out/'development-selection.json').read_text(encoding='utf-8'))
    if (s['records_sha256']!=manifest['records_sha256'] or s['data_commit']!=commit
            or s['protocol_commit']!=PROTOCOL_COMMIT):raise CorpusError('Desarrollo de otro corpus/protocolo.')
    observed=_require_unobserved(manifest['records_sha256'],reproduce_observed=reproduce_observed)
    verify_selection_commit(s,selection_commit,reproduce_observed=reproduce_observed)
    if (out/'assessment-started.json').exists():raise CorpusError('Assessment observado/iniciado; conservarlo y usar otro destino para reproducir.')
    cap=min(900,p['compute_budget']['max_replica_and_assessment_wall_seconds']);deadline=start+cap
    scope=REPRODUCTION_SCOPE if reproduce_observed else FRESH_SCOPE
    entries=list(s['model_entries']);replica_status=[];policies={};results={};saved=[];locked=False
    evidence={'assessment_prediction_calls_started':0,'assessment_prediction_calls_returned':0}
    def progress(stage,**values):
        evidence['stage']=stage
        _snapshot(out/'assessment-progress.json',{'stage':stage,'elapsed_seconds':time.perf_counter()-start,
                  'scope':scope,**values})
        print(json.dumps({'stage':stage,**values}),flush=True)
    try:
        _check_deadline(deadline,'assessment_setup')
        if replicas and s['selected_adjusted']:
            for seed in p['replica_seeds']:
                # Reserve calibration/evaluation time; never drop a seed due to score.
                _check_deadline(deadline,'replica_planning')
                estimated=s['selected_adjusted']['training']['fit_wall_seconds']+30
                if time.perf_counter()+estimated+90>deadline:
                    replica_status.append({'seed':seed,'status':'NOT_EXECUTED_BUDGET','estimated_fit_seconds':estimated});continue
                folder=out/'replicas'/str(seed)
                try:
                    replay(folder,registration,cache,commit,out/'selected-configuration.json',seed,deadline=deadline-90)
                    _check_deadline(deadline,'replica_complete')
                    entries.append(_entry(out,'beto_seed'+str(seed),'beto',artifact=folder/'model',training={'fixed_seed':seed,'selection_retuned':False}))
                    replica_status.append({'seed':seed,'status':'EXECUTED'})
                except BudgetExceeded as exc:replica_status.append({'seed':seed,'status':'INTERRUPTED_BUDGET','stage':exc.stage})
        _bounded_call(deadline,'evaluation_models_write',write_json,out/'evaluation-models.json',entries)
        _bounded_call(deadline,'replica_status_write',write_json,out/'replica-status.json',replica_status)
        for entry in entries:
            key=entry['key'];progress('CALIBRATION_POLICY_ONLY',model=key)
            model=_bounded_call(deadline,'calibration_load:'+key,_load_entry,out,entry)
            try:
                cal=_score_with_deadline(model,[r['text'] for r in parts['calibration']],deadline,'calibration:'+key,evidence=evidence)
                t=_bounded_call(deadline,'temperature_selection:'+key,fit_temperature,parts['calibration'],cal,p)
                pol=temperature_transform(_score_with_deadline(model,[r['text'] for r in parts['policy_dev']],
                    deadline,'policy_dev:'+key,evidence=evidence),t['temperature'])
                amb=temperature_transform(_score_with_deadline(model,[r['text'] for r in parts['ambiguity_policy']],
                    deadline,'ambiguity_policy:'+key,evidence=evidence),t['temperature'])
                policy=_bounded_call(deadline,'policy_selection:'+key,select_policy,parts['policy_dev'],pol,parts['ambiguity_policy'],amb,p)
                policies[key]={'temperature':t,'policy':policy}
            finally:del model;gc.collect()
        _bounded_call(deadline,'calibration_policy_write',write_json,out/'calibration-policy-selection.json',policies)
        if json.loads((out/'development-selection.json').read_text(encoding='utf-8'))!=s:
            raise CorpusError('La selección cambió durante calibración; no predecir assessment.')
        # Every configuration, seed artifact and policy choice is present before
        # the corpus-wide exclusive marker, itself before any final prediction.
        _check_deadline(deadline,'assessment_lock:before')
        acquire_assessment_lock(out,{'records_sha256':manifest['records_sha256'],'selection_commit':selection_commit,
            'selection_sha256':file_sha(out/'development-selection.json'),'policy_sha256':file_sha(out/'calibration-policy-selection.json'),
            'models_sha256':file_sha(out/'evaluation-models.json'),'scope':scope},reproduce_observed=reproduce_observed)
        locked=True;_check_deadline(deadline,'assessment_lock:after')
        for entry in entries:
            key=entry['key'];progress('ASSESSMENT_LOCKED',model=key)
            model=_bounded_call(deadline,'assessment_load:'+key,_load_entry,out,entry)
            try:
                rows=parts['assessment'];texts=[r['text'] for r in rows]
                begin=time.perf_counter()
                raw=_score_with_deadline(model,texts,deadline,'assessment:'+key,evidence=evidence,assessment=True)
                seconds=time.perf_counter()-begin
                truth=np.array([r['intended_label'] for r in rows]);guess=np.array(LABELS)[raw.argmax(axis=1)]
                legacy=review_mask(raw,texts);temp=policies[key]['temperature']['temperature'];cal=temperature_transform(raw,temp)
                amb=temperature_transform(_score_with_deadline(model,[r['text'] for r in parts['ambiguity_assessment']],
                    deadline,'ambiguity_assessment:'+key,evidence=evidence,assessment=True),temp)
                selected_policy=policies[key]['policy']['selected']
                _check_deadline(deadline,'assessment_metrics:'+key+':before')
                selective=describe_policy(rows,cal,parts['ambiguity_assessment'],amb,selected_policy,p['policy']) if selected_policy else None
                metrics={'classification':summarize_predictions(truth,guess,reviewed=legacy),
                    'slices':slice_metrics(rows,truth,guess,legacy),'raw_probability_quality':probability_metrics(rows,raw,5),
                    'calibrated_probability_quality':probability_metrics(rows,cal,5),'selected_policy_assessment':selective}
                _check_deadline(deadline,'assessment_metrics:'+key+':after')
                timing=[]
                _inference_call(deadline,'latency_warmup:'+key,model.predict_proba,[parts['tune'][0]['text']],evidence=evidence)
                for i,row in enumerate(parts['tune'][:12]):
                    t0=time.perf_counter_ns()
                    _inference_call(deadline,f'latency:{key}:{i}',model.predict_proba,[row['text']],evidence=evidence)
                    timing.append((time.perf_counter_ns()-t0)/1e6)
                results[key]={**metrics,
                    'latency':{'assessment_seconds':seconds,'rows':len(rows),'warm_batch1_ms':timing,
                        'p50_ms':float(np.percentile(timing,50)),'p95_ms':float(np.percentile(timing,95)),
                        'scope':'12 warm tune requests, same texts, loaded backend includes vectorization/tokenization; not production-tail estimate'},
                    'errors':[{'review_id':r['review_id'],'family':r['family'],'text':r['text'],'intended_label':str(y),'prediction':str(g)}
                               for r,y,g in zip(rows,truth,guess) if y!=g],
                    'inference_memory_scope':'separate stdlib-parent inference-memory.json, never optimizer peak'}
                saved += [{'model':key,'review_id':r['review_id'],'family':r['family'],'intended_label':str(y),'prediction':str(g),
                           'probabilities':prob.tolist(),'calibrated_probabilities':cp.tolist()} for r,y,g,prob,cp in zip(rows,truth,guess,raw,cal)]
                _check_deadline(deadline,'assessment_model_complete:'+key)
            finally:del model;gc.collect()
        report={'schema':'beto-v3-pilot-assessment','status':'COMPLETED_WITHIN_REGISTERED_BUDGET',
            'data_manifest':manifest,'data_commit':commit,'protocol_commit':PROTOCOL_COMMIT,'selection_commit':selection_commit,
            'selection_sha256':file_sha(out/'development-selection.json'),'policy_sha256':file_sha(out/'calibration-policy-selection.json'),
            'selected_model_key':s['selected_model_key'],'best_development_candidate':s['best_overall']['id'],
            'results':results,'replica_status':replica_status,'model_entries':entries,'scope':scope,'prior_observation_evidence':observed,
            'cost':{'assessment_and_replica_wall_seconds':time.perf_counter()-start,'wall_seconds_cap':cap,**resource_report()},
            'human_performance_measured':False,'new_assessment_is_now_observed':not reproduce_observed,
            'limits':p['limits']+['AI-reviewed synthetic intent, not human gold. Historical-recipe references retrained on NEW TRAIN, not original V1/V2 binaries.',
                 'Pilot rows do not count as 5000 achieved; full-scale final must be newly generated and frozen.',
                 'No policy or model is promoted automatically, including an apparently best seed or recipe.']}
        _bounded_call(deadline,'assessment_predictions_write',write_json,out/'assessment-predictions.json',saved)
        _bounded_call(deadline,'assessment_report_write',write_json,out/'assessment.json',report)
        return report
    except BaseException as exc:
        # Preserve a conservative observation reservation even if an inference
        # returned after the deadline, or a later model/latency call failed.
        if (out/'assessment.json').exists():(out/'assessment.json').rename(out/'assessment-interrupted-report.json')
        write_json(out/'assessment-interruption.json',{**evidence,'status':'INTERRUPTED_COMPUTE_BUDGET' if isinstance(exc,BudgetExceeded) else 'INTERRUPTED_ERROR',
            'stage':getattr(exc,'stage',evidence.get('stage','assessment_setup')),'error_type':type(exc).__name__,
            'scope':scope,'records_sha256':manifest['records_sha256'],'selection_commit':selection_commit,
            'assessment_lock_retained':locked,'assessment_predictions_attempted':evidence['assessment_prediction_calls_started']>0,
            'corpus_observation_marker_retained':_observation_marker(manifest['records_sha256']).exists(),
            'wall_seconds':time.perf_counter()-start,'wall_seconds_cap':cap,'completed_model_keys':list(results),
            'replica_status':replica_status})
        if results or saved:write_json(out/'assessment-partial-results.json',{'results':results,'predictions':saved,'complete':False})
        raise


def measure(out,key):
    entries=json.loads((out/'evaluation-models.json').read_text(encoding='utf-8'))
    entry=next(e for e in entries if e['key']==key);model=_load_entry(out,entry)
    parts,_,_=data.load_data(require_review=True)
    for row in parts['tune'][:9]:model.predict_proba([row['text']])
    return {'model':key,**resource_report(),'scope':'isolated inference process, imports/load/nine tune requests, no assessment calls'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('audit','blind-packet','register','develop','replay','assess','measure'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--registration',type=Path,default=Path('artifacts/beto-v3-registration'))
    parser.add_argument('--cache',type=Path,default=Path('artifacts/beto-v3-checkpoint'));parser.add_argument('--registration-commit')
    parser.add_argument('--selection',type=Path);parser.add_argument('--seed',type=int,default=42);parser.add_argument('--model-key')
    parser.add_argument('--selection-commit',help='Commit completo que contiene docs/research/beto-v3-selection.json antes de assessment.')
    parser.add_argument('--reproduce-observed',action='store_true',help='Reproducir decisiones registradas; nunca presentar el corpus observado como un test nuevo.')
    parser.add_argument('--download',action='store_true');args=parser.parse_args(argv)
    if args.phase=='audit':
        out=ignored_target(args.output,kinds=('artifacts',));report=data.automatic_audit(args.cache,download=args.download);write_json(out,report)
        print(json.dumps({'automatic_gates_pass':report['automatic_gates_pass'],'checks':report['checks']}));return
    if args.phase=='blind-packet':
        _,n=data.prepare_blind_packet(args.output);print(json.dumps({'blind_items':n}));return
    if args.phase=='measure':print(json.dumps(measure(args.output,args.model_key)));return
    if args.phase=='register':print(json.dumps({'registered':register(args.output,args.registration_commit)['records_sha256']}));return
    if args.phase=='develop':
        r=develop(args.output,args.registration,args.cache,args.registration_commit,download=args.download,reproduce_observed=args.reproduce_observed)
        print(json.dumps({'selected':r['selected_model_key'],'cost':r['cost']}));return
    if args.phase=='replay':
        r=replay(args.output,args.registration,args.cache,args.registration_commit,args.selection,args.seed,download=args.download)
        print(json.dumps({'seed':r['seed'],'assessment_used':r['assessment_used'],'platform':r['platform']}));return
    if not args.selection_commit:parser.error('assess requiere --selection-commit antes de predecir assessment.')
    r=assess(args.output,args.registration,args.cache,args.registration_commit,
             selection_commit=args.selection_commit,reproduce_observed=args.reproduce_observed)
    print(json.dumps({'macro_f1':{k:v['classification']['macro_f1'] for k,v in r['results'].items()},'cost':r['cost']}))


if __name__=='__main__':main()
