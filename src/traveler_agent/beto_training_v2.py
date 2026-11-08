"""BETO Training V2: registered development, separate assessment and Windows replay."""
from __future__ import annotations

import argparse
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import joblib
import numpy as np

from .beto_classifier import BetoClassifier
from .beto_training_v2_data import load_data,register,verify_registration,training_diagnostics,CATALOGS,PROTOCOL
from .beto_training_v2_model import EncoderV2,fit_head,train_encoder,export_model,load_model_v2,tuning_score,score_rank
from . import beto_training_v2_policy as policy_study
from .contextual_benchmark import train_baselines,resource_report
from .corpus import CorpusError,file_sha
from .labels import LABELS
from .research_metrics import score_matrix,summarize_predictions,slice_metrics,review_mask
from .research_v3_selective import fit_temperature,temperature_transform,probability_metrics
from .synthetic_diversity import ignored_target,write_json,canonical_sha


def verify_git_registration(commit):
    if not commit or not re.fullmatch(r'[0-9a-f]{40}',commit):raise CorpusError('Commit de registro completo requerido.')
    for path in (*CATALOGS,PROTOCOL,PROTOCOL.with_name('synthetic_diversity_v3.json')):
        name='src/traveler_agent/research_data/'+path.name
        content=subprocess.check_output(['git','show',f'{commit}:{name}'],stderr=subprocess.DEVNULL)
        if hashlib.sha256(content).hexdigest()!=file_sha(path):raise CorpusError('Registro Git no coincide con datos/protocolo.')


def arms(parts):
    return {'existing120':[r for r in parts['train'] if r['source_arm']=='existing120'],
            'new120':[r for r in parts['train'] if r['source_arm']=='new120'],'mixed240':parts['train']}


def _entry(output,key,kind,*,artifact=None,pipeline=None,configuration=None,training=None):
    record={'key':key,'kind':kind,'training':training or {}}
    if artifact:record['artifact']=str(artifact.relative_to(output));record['manifest_sha256']=file_sha(artifact/'model_manifest.json')
    if pipeline is not None:
        path=output/'classical'/f'{key}.joblib';path.parent.mkdir(exist_ok=True);joblib.dump(pipeline,path)
        record.update(artifact=str(path.relative_to(output)),sha256=file_sha(path))
    if configuration:record['configuration']=configuration
    return record


def _load_entry(output,entry):
    path=ignored_target(output/entry['artifact'],kinds=('artifacts',))
    if entry['kind']=='beto':
        if file_sha(path/'model_manifest.json')!=entry['manifest_sha256']:raise CorpusError('Manifiesto de modelo cambiado.')
        return load_model_v2(path)
    if file_sha(path)!=entry['sha256']:raise CorpusError('Pipeline propio modificado.')
    return joblib.load(path)  # Only this run's locally created, hashed pipelines.


def develop(output,registration,cache,commit,*,download=False):
    verify_git_registration(commit);parts,manifest,protocol=verify_registration(registration)
    output=ignored_target(output,kinds=('artifacts',))
    if output.exists():raise CorpusError('Destino existente; no sobrescribir desarrollo/evaluación.')
    output.mkdir(parents=True);write_json(output/'data-manifest.json',manifest);write_json(output/'protocol.json',protocol)
    write_json(output/'training-diagnostics.json',training_diagnostics(parts['train']))
    seed=protocol['primary_seed'];start=time.perf_counter();entries=[];records={};training=arms(parts)
    def progress(stage,**values):
        value={'stage':stage,'elapsed_seconds':time.perf_counter()-start,**values}
        (output/'progress.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(value,ensure_ascii=False),flush=True)
    progress('BASELINES')
    baseline_protocol={'seed':seed,'baselines':{'lr_C':protocol['baseline_C'],'svm_C':protocol['baseline_C']}}
    for arm,rows in training.items():
        models,scores=train_baselines(rows,parts['tune'],baseline_protocol)
        for kind,model in models.items():
            if kind=='dummy' and arm!='mixed240':continue
            key=f'{kind}_{arm}';entries.append(_entry(output,key,'classical',pipeline=model,training={'arm':arm,'rows':len(rows)}));records[key]=scores[kind]
        del models;gc.collect()
    head_cache={};head_logs={};frozen_entries={};feature_statistics={}
    # All feature extraction here is strictly TRAIN/tune. No final embeddings.
    for pooling,length in (('mean_l2',128),('cls_l2',128),('cls_l2',256)):
        progress('HEAD_FEATURES',pooling=pooling,max_length=length)
        encoder=EncoderV2(cache,download=download,pooling=pooling,max_length=length,threads=protocol['threads'])
        train_features=encoder.encode([r['text'] for r in parts['train']]);tune_features=encoder.encode([r['text'] for r in parts['tune']])
        token_counts=[len(encoder.tokenizer(r['text'],truncation=False)['input_ids']) for r in parts['train']]
        feature_statistics[f'{pooling}_{length}']={'train_rows_truncated':sum(n>length for n in token_counts),'train_rows':len(token_counts),
                                                'max_untruncated_train_tokens':max(token_counts)}
        selected_arms=training if (pooling,length)==('mean_l2',128) else {'mixed240':parts['train']}
        positions={r['review_id']:i for i,r in enumerate(parts['train'])}
        for arm,rows in selected_arms.items():
            indices=[positions[r['review_id']] for r in rows]
            weights,log=fit_head(train_features[indices],rows,tune_features,parts['tune'],protocol,seed)
            id_=f'{pooling}_{length}_{arm}';head_cache[id_]=weights;head_logs[id_]=log
            path=output/'models'/('frozen_'+id_)
            config={'mode':'frozen','pooling':pooling,'max_length':length}
            export_model(path,cache,weights,config,{'supervised':True,'mode':'frozen_head_v2','seed':seed,'arm':arm,
                'head_epoch':log['selected']['epoch'],'records_sha256':manifest['records_sha256'],'registration_commit':commit})
            entry=_entry(output,'beto_frozen_'+id_,'beto',artifact=path,configuration=config,training={'arm':arm,'head':log})
            frozen_entries[id_]=entry
            if pooling=='mean_l2' and length==128:entries.append(entry)
        del encoder,train_features,tune_features;gc.collect()
    candidates=[];candidate_entries={}
    for config in protocol['candidates']:
        progress('CANDIDATE_START',candidate=config['id'],memory=resource_report())
        if time.perf_counter()-start>protocol['compute_budget']['max_development_wall_seconds']:
            candidates.append({'id':config['id'],'status':'NOT_EXECUTED_COMPUTE_BUDGET'});continue
        pool=config['pooling'];length=config['max_length'];head_id=f'{pool}_{length}_mixed240'
        if config['mode']=='frozen':
            entry=frozen_entries[head_id];log=head_logs[head_id]
            score=log['selected']['tune'];candidate_entries[config['id']]=entry
            candidates.append({'id':config['id'],'status':'EXECUTED','configuration':config,'head_epoch':log['selected']['epoch'],
                               'selected_epoch':0,'tune':score,'training':log});continue
        encoder=EncoderV2(cache,pooling=pool,max_length=length,threads=protocol['threads'])
        state=output/'states'/f'{config["id"]}.safetensors'
        weights,log=train_encoder(encoder,parts['train'],parts['tune'],head_cache[head_id],config,protocol,seed,state)
        artifact=output/'models'/config['id']
        export_model(artifact,cache,weights,config,{'supervised':True,'mode':'staged_'+config['mode']+'_v2','seed':seed,
            'head_epoch':head_logs[head_id]['selected']['epoch'],'encoder_epoch':log['selected']['epoch'],
            'records_sha256':manifest['records_sha256'],'registration_commit':commit},state=state)
        entry=_entry(output,'beto_'+config['id'],'beto',artifact=artifact,configuration=config,training={'head':head_logs[head_id],'encoder':log})
        candidate_entries[config['id']]=entry
        candidates.append({'id':config['id'],'status':'EXECUTED','configuration':config,'head_epoch':head_logs[head_id]['selected']['epoch'],
                           'selected_epoch':log['selected']['epoch'],'tune':log['selected']['tune'],'training':log})
        progress('CANDIDATE_FINISHED',candidate=config['id'],epoch=log['selected']['epoch'],tune=log['selected']['tune'],memory=resource_report())
        del encoder,weights;gc.collect()
    executed=[r for r in candidates if r['status']=='EXECUTED']
    chosen=max(executed,key=lambda row:score_rank(row['tune']))
    selected_entry=candidate_entries[chosen['id']]
    if selected_entry['key'] not in {e['key'] for e in entries}:entries.append(selected_entry)
    selection={'schema':'beto-training-v2-development','registration_commit':commit,'records_sha256':manifest['records_sha256'],
        'protocol_sha256':manifest['protocol_sha256'],'primary_seed':seed,'classical_selection':records,'head_selection':head_logs,
        'candidates':candidates,'selected_candidate':chosen,'selected_model_key':selected_entry['key'],
        'feature_statistics':feature_statistics,'model_entries':entries,'assessment_predictions_executed':False,
        'cost':{'wall_seconds':time.perf_counter()-start,**resource_report()},
        'environment':{'python':platform.python_version(),'platform':platform.platform(),'torch':version('torch'),'transformers':version('transformers')}}
    write_json(output/'development-selection.json',selection)
    # A standalone file can be supplied to a Windows job. No final labels/results.
    write_json(output/'selected-configuration.json',{k:selection[k] for k in
        ('schema','registration_commit','records_sha256','protocol_sha256','primary_seed','selected_candidate','selected_model_key')})
    progress('DEVELOPMENT_COMPLETED',selected=chosen['id'])
    return selection


def replay_selected(output,registration,cache,commit,selection_file,seed,*,download=False):
    verify_git_registration(commit);parts,manifest,protocol=verify_registration(registration)
    selection=json.loads(selection_file.read_text(encoding='utf-8'))
    if (selection['records_sha256']!=manifest['records_sha256'] or selection['registration_commit']!=commit
            or selection['protocol_sha256']!=manifest['protocol_sha256']):
        raise CorpusError('Selección de otro registro.')
    output=ignored_target(output,kinds=('artifacts',))
    if output.exists():raise CorpusError('Destino de réplica existente.')
    output.mkdir(parents=True);chosen=selection['selected_candidate'];config=chosen['configuration']
    encoder=EncoderV2(cache,download=download,pooling=config['pooling'],max_length=config['max_length'],threads=protocol['threads'])
    x=encoder.encode([r['text'] for r in parts['train']]);dev=encoder.encode([r['text'] for r in parts['tune']])
    weights,head_log=fit_head(x,parts['train'],dev,parts['tune'],protocol,seed,fixed_epoch=chosen['head_epoch'])
    state=None;log=None
    if config['mode']!='frozen':
        state=output/'state.safetensors'
        weights,log=train_encoder(encoder,parts['train'],parts['tune'],weights,config,protocol,seed,state,fixed_epoch=chosen['selected_epoch'])
    export_model(output/'model',cache,weights,config,{'supervised':True,'mode':'fixed_selected_'+config['mode']+'_v2','seed':seed,
        'head_epoch':chosen['head_epoch'],'encoder_epoch':chosen['selected_epoch'],'registration_commit':commit,
        'records_sha256':manifest['records_sha256'],'selection_retuned':False},state=state)
    result={'seed':seed,'head_training':head_log,'encoder_training':log,'fixed_configuration':chosen,'assessment_used':False,
            'artifact':str(output/'model'),'memory':resource_report(),'platform':platform.platform()}
    write_json(output/'replay.json',result);del encoder;gc.collect()
    return result


def assess(output,registration,cache,commit,*,replicas=True):
    verify_git_registration(commit);parts,manifest,protocol=verify_registration(registration)
    output=ignored_target(output,kinds=('artifacts',))
    selection=json.loads((output/'development-selection.json').read_text(encoding='utf-8'))
    if selection['records_sha256']!=manifest['records_sha256']:raise CorpusError('Desarrollo de otro catálogo.')
    if (output/'assessment.json').exists():raise CorpusError('Assessment ya observado; conservar resultados.')
    entries=list(selection['model_entries']);start=time.perf_counter();predictions=[];results={};policies={}
    if replicas:
        for seed in protocol['replica_seeds']:
            folder=output/'replicas'/str(seed)
            replay_selected(folder,registration,cache,commit,output/'selected-configuration.json',seed)
            entry=_entry(output,f'beto_selected_seed{seed}','beto',artifact=folder/'model',training={'seed':seed,'retuned':False})
            entries.append(entry)
    write_json(output/'evaluation-models.json',entries)
    # Calibration and policy choices for EVERY entry are fixed before assessment.
    for entry in entries:
        model=_load_entry(output,entry);key=entry['key']
        cal=score_matrix(model,[r['text'] for r in parts['calibration']])
        temperature=fit_temperature(parts['calibration'],cal,protocol)
        pol=temperature_transform(score_matrix(model,[r['text'] for r in parts['policy_dev']]),temperature['temperature'])
        amb=temperature_transform(score_matrix(model,[r['text'] for r in parts['ambiguity_policy']]),temperature['temperature'])
        policies[key]={'temperature':temperature,'policy':policy_study.select(parts['policy_dev'],pol,parts['ambiguity_policy'],amb,protocol)}
        del model;gc.collect()
    write_json(output/'calibration-policy-selection.json',policies)
    selection_sha=file_sha(output/'development-selection.json');policy_sha=file_sha(output/'calibration-policy-selection.json')
    for entry in entries:
        model=_load_entry(output,entry);key=entry['key'];rows=parts['assessment'];texts=[r['text'] for r in rows]
        begin=time.perf_counter();raw=score_matrix(model,texts);seconds=time.perf_counter()-begin
        prediction=np.asarray(LABELS)[raw.argmax(axis=1)];truth=np.asarray([r['intended_label'] for r in rows]);legacy=review_mask(raw,texts)
        t=policies[key]['temperature']['temperature'];calibrated=temperature_transform(raw,t)
        ambiguous=score_matrix(model,[r['text'] for r in parts['ambiguity_assessment']])
        selected_policy=policies[key]['policy']['selected']
        policy_result=policy_study.describe(rows,calibrated,parts['ambiguity_assessment'],temperature_transform(ambiguous,t),selected_policy,protocol['policy']) if selected_policy else None
        timings=[];model.predict_proba([parts['tune'][0]['text']])
        for row in parts['tune'][:8]:
            begin=time.perf_counter_ns();model.predict_proba([row['text']]);timings.append((time.perf_counter_ns()-begin)/1e6)
        results[key]={'classification':summarize_predictions(truth,prediction,reviewed=legacy),
            'slices':slice_metrics(rows,truth,prediction,legacy),'raw_probability_quality':probability_metrics(rows,raw,5),
            'calibrated_probability_quality':probability_metrics(rows,calibrated,5),'selected_policy_assessment':policy_result,
            'latency':{'assessment_seconds':seconds,'rows':len(rows),'warm_batch1_ms':timings,'p50_ms':float(np.percentile(timings,50)),
                       'p95_ms':float(np.percentile(timings,95)),'scope':'8 warm tune requests; loaded model includes tokenization/vectorization'},
            'errors':[{'review_id':r['review_id'],'family':r['family'],'text':r['text'],'intended_label':str(y),'prediction':str(p)}
                      for r,y,p in zip(rows,truth,prediction) if y!=p]}
        for row,y,p,s,c in zip(rows,truth,prediction,raw,calibrated):
            predictions.append({'model':key,'review_id':row['review_id'],'family':row['family'],'intended_label':str(y),
                                'prediction':str(p),'probabilities':s.tolist(),'calibrated_probabilities':c.tolist()})
        del model;gc.collect()
        # Linux child high-water marks can inherit the optimizer parent's peak.
        # Measure from scripts/measure_beto_v2_memory.py in a separate, stdlib-only
        # launcher after this process exits; do not attribute this peak to a model.
        results[key]['inference_memory_scope']='See separate inference-memory.json; not this optimizer process peak.'
    report={'schema':'beto-training-v2-assessment','registration_commit':commit,'data_manifest':manifest,
        'development_selection_sha256':selection_sha,'calibration_policy_selection_sha256':policy_sha,
        'selected_model_key':selection['selected_model_key'],'results':results,'model_entries':entries,
        'cost':{'assessment_and_replica_wall_seconds':time.perf_counter()-start,**resource_report()},
        'limits':protocol['limits'],'new_assessment_is_now_observed':True,'human_performance_measured':False}
    write_json(output/'assessment-predictions.json',predictions);write_json(output/'assessment.json',report)
    return report


def measure_inference(output,key):
    entries=json.loads((output/'evaluation-models.json').read_text())
    entry=next(row for row in entries if row['key']==key)
    model=_load_entry(output,entry);parts,_,_=load_data()
    model.predict_proba([parts['tune'][0]['text']])
    for row in parts['tune'][:8]:model.predict_proba([row['text']])
    return {'model':key,**resource_report(),'scope':'fresh process; imports, load and 9 tune inferences; no assessment calls'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','develop','assess','replay','measure'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--registration',type=Path,default=Path('artifacts/beto-v2-registration'))
    parser.add_argument('--cache',type=Path,default=Path('artifacts/beto-v2-checkpoint'));parser.add_argument('--registration-commit')
    parser.add_argument('--download',action='store_true');parser.add_argument('--selection',type=Path);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--model-key')
    args=parser.parse_args(argv)
    if args.phase=='measure':print(json.dumps(measure_inference(args.output,args.model_key)));return
    if args.phase=='register':r=register(args.output);print(json.dumps({'registered':r['records_sha256'],'counts':{k:v['rows'] for k,v in r['counts'].items()}}));return
    if args.phase=='develop':r=develop(args.output,args.registration,args.cache,args.registration_commit,download=args.download);print(json.dumps({'selected':r['selected_candidate'],'cost':r['cost']},default=str));return
    if args.phase=='replay':r=replay_selected(args.output,args.registration,args.cache,args.registration_commit,args.selection,args.seed,download=args.download);print(json.dumps({'platform':r['platform'],'seed':r['seed'],'assessment_used':False}));return
    r=assess(args.output,args.registration,args.cache,args.registration_commit)
    print(json.dumps({'macro_f1':{k:v['classification']['macro_f1'] for k,v in r['results'].items()},'cost':r['cost']},indent=2))


if __name__=='__main__':main()
