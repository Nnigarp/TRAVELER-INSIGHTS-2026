"""Finish missing fixed inference after a disclosed budget stop; never refit."""
from __future__ import annotations
import argparse,gc,json,math,os,shutil,time
from pathlib import Path
import numpy as np
from . import beto_training_v3 as base
from .corpus import CorpusError,file_sha
from .synthetic_diversity import ignored_target,write_json,canonical_sha

STATUS='COMPLETED_WITH_DISCLOSED_FIXED_INFERENCE_CONTINUATION'
PROTOCOL_PATH='docs/research/beto-v3-inference-continuation-protocol.json'
ALIASES={'assessment-interruption.json':'prior-assessment-interruption.json',
         'assessment-partial-results.json':'prior-assessment-partial-results.json',
         'assessment-started.json':'prior-assessment-started.json'}


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def verify_prefix(members,entries,partial,interruption):
    completed=set(partial['results']);ordered=[e['key'] for e in entries]
    prefix=ordered[:len(completed)]
    if (partial.get('complete') is not False or completed!=set(prefix)
            or completed!=set(interruption['completed_model_keys'])):
        raise CorpusError('The completed original model prefix changed.')
    saved=partial['predictions'];n=len(members)
    if len(saved)!=n*len(prefix):raise CorpusError('Incomplete original prediction prefix.')
    for i,key in enumerate(prefix):
        rows=saved[i*n:(i+1)*n]
        for actual,member in zip(rows,members):
            if actual['model']!=key or any(actual[f]!=member[f] for f in ('review_id','family','intended_label')):
                raise CorpusError('Original prediction membership changed.')
            p=np.asarray(actual['probabilities'],dtype=float)
            if (p.shape!=(3,) or not np.isfinite(p).all() or p.min()<0 or not np.isclose(p.sum(),1)
                    or actual['prediction']!=base.LABELS[int(p.argmax())]):
                raise CorpusError('Invalid original raw prediction.')
        calculated=base.summarize_predictions([r['intended_label'] for r in rows],[r['prediction'] for r in rows])
        declared=partial['results'][key]['classification']
        if any(declared[f]!=calculated[f] for f in ('n','support','confusion_matrix','macro_f1','negative_to_positive_count')):
            raise CorpusError('Original prefix metrics do not match its raw predictions.')
    return prefix


def verify_sources(root,protocol,*,copied=False):
    required={'assessment-interruption.json','assessment-partial-results.json','assessment-started.json',
              'development-selection.json','calibration-policy-selection.json','evaluation-models.json',
              'replica-status.json','manifest.json','protocol.json'}
    if set(protocol.get('source_sha256',{}))!=required:raise CorpusError('Incomplete continuation source hash set.')
    for name,digest in protocol['source_sha256'].items():
        path=root/(ALIASES.get(name,name) if copied else name)
        if file_sha(path)!=digest:raise CorpusError('Locked continuation source changed: '+name)
    interrupted=read(root/(ALIASES['assessment-interruption.json'] if copied else 'assessment-interruption.json'))
    lock=read(root/(ALIASES['assessment-started.json'] if copied else 'assessment-started.json'))
    if (interrupted['status']!='INTERRUPTED_COMPUTE_BUDGET' or not interrupted['assessment_lock_retained']
            or lock['scope']!=base.REPRODUCTION_SCOPE
            or lock['records_sha256']!=protocol['records_sha256']
            or lock['policy_sha256']!=file_sha(root/'calibration-policy-selection.json')
            or lock['models_sha256']!=file_sha(root/'evaluation-models.json')
            or lock['selection_sha256']!=file_sha(root/'development-selection.json')):
        raise CorpusError('Original policy/model observation lock is incomplete or changed.')
    if (protocol.get('registered_after_original_budget_interruption') is not True
            or protocol.get('maximum_additional_inference_seconds')!=300
            or protocol.get('unchanged_combined_cap_seconds')!=1800):
        raise CorpusError('Disclosed inference-only allocation required.')
    return interrupted,lock


def evaluate_one(model,key,parts,policy,p,deadline):
    rows=parts['assessment'];texts=[r['text'] for r in rows];start=time.perf_counter()
    raw=base._score_with_deadline(model,texts,deadline,'continued_assessment:'+key)
    seconds=time.perf_counter()-start;truth=np.asarray([r['intended_label'] for r in rows])
    guess=np.asarray(base.LABELS)[raw.argmax(axis=1)];legacy=base.review_mask(raw,texts)
    temperature=policy['temperature']['temperature'];cal=base.temperature_transform(raw,temperature)
    amb=base.temperature_transform(base._score_with_deadline(model,[r['text'] for r in parts['ambiguity_assessment']],deadline,'continued_ambiguity:'+key),temperature)
    choice=policy['policy']['selected']
    selective=base.describe_policy(rows,cal,parts['ambiguity_assessment'],amb,choice,p['policy']) if choice else None
    timing=[];base._inference_call(deadline,'continued_warmup',model.predict_proba,[parts['tune'][0]['text']])
    for row in parts['tune'][:12]:
        begin=time.perf_counter_ns();base._inference_call(deadline,'continued_latency',model.predict_proba,[row['text']])
        timing.append((time.perf_counter_ns()-begin)/1e6)
    result={'classification':base.summarize_predictions(truth,guess,reviewed=legacy),
        'slices':base.slice_metrics(rows,truth,guess,legacy),
        'raw_probability_quality':base.probability_metrics(rows,raw,5),
        'calibrated_probability_quality':base.probability_metrics(rows,cal,5),
        'selected_policy_assessment':selective,
        'latency':{'assessment_seconds':seconds,'rows':len(rows),'warm_batch1_ms':timing,
            'p50_ms':float(np.percentile(timing,50)),'p95_ms':float(np.percentile(timing,95)),
            'scope':'same 12 warm tune requests, batch1, tokenization/vectorization included; not a production-tail estimate'},
        'errors':[{'review_id':r['review_id'],'family':r['family'],'text':r['text'],'intended_label':str(y),'prediction':str(g)}
                  for r,y,g in zip(rows,truth,guess) if y!=g],
        'inference_memory_scope':'separate stdlib-parent inference-memory.json, never optimizer peak'}
    saved=[{'model':key,'review_id':r['review_id'],'family':r['family'],'intended_label':str(y),'prediction':str(g),
            'probabilities':v.tolist(),'calibrated_probabilities':c.tolist()}
           for r,y,g,v,c in zip(rows,truth,guess,raw,cal)]
    base._check_deadline(deadline,'continued_model_complete');return result,saved


def validate_ledger(root,report,saved):
    """Saved-only validation: no corpus access, fitting, prediction or Git call."""
    if not isinstance(saved,list):raise CorpusError('Saved continuation predictions must be a list.')
    if not (root/'continuation-protocol.json').exists():raise CorpusError('Missing disclosed continuation proof.')
    ledger=report.get('continuation',{});protocol=read(root/'continuation-protocol.json')
    interrupted,_=verify_sources(root,protocol,copied=True)
    partial=read(root/'prior-assessment-partial-results.json');entries=read(root/'evaluation-models.json')
    if set(report['results'])!={e['key'] for e in entries}:raise CorpusError('The fixed continuation is still incomplete.')
    verify_prefix(report['data_manifest']['assessment_membership'],entries,partial,interrupted)
    if (report.get('status')!=STATUS or report.get('scope')!=base.REPRODUCTION_SCOPE
            or ledger.get('protocol_sha256')!=file_sha(root/'continuation-protocol.json')
            or saved[:len(partial['predictions'])]!=partial['predictions']
            or any(report['results'].get(k)!=v for k,v in partial['results'].items())):
        raise CorpusError('Completed prefix or disclosed continuation proof changed.')
    duration=ledger.get('wall_seconds');prep=protocol['preparation_seconds'];first=interrupted['wall_seconds']
    if (isinstance(duration,bool) or not isinstance(duration,(int,float)) or not math.isfinite(duration)
            or not 0<=duration<=300 or prep+first+duration>1800
            or abs(report['cost']['assessment_and_replica_wall_seconds']-(first+duration))>1e-6
            or ledger.get('new_training_calibration_or_selection') is not False):
        raise CorpusError('Continuation compute ledger is invalid or exceeds the total budget.')
    return True


def finish(source,out,registration_commit):
    start=time.perf_counter();source=ignored_target(source,kinds=('artifacts',));out=ignored_target(out,kinds=('artifacts',))
    protocol=base._git_json(registration_commit,PROTOCOL_PATH)
    if protocol!=read(Path(PROTOCOL_PATH)):raise CorpusError('Continuation protocol differs from its Git registration.')
    interrupted,lock=verify_sources(source,protocol);selection=read(source/'development-selection.json')
    base.verify_selection_commit(selection,protocol['original_selection_commit'],reproduce_observed=True)
    parts,manifest,p=base.verify(source/'registration',selection['data_commit'])
    if manifest['records_sha256']!=protocol['records_sha256']:raise CorpusError('Continuation of another corpus.')
    partial=read(source/'assessment-partial-results.json');entries=read(source/'evaluation-models.json')
    prefix=verify_prefix(manifest['assessment_membership'],entries,partial,interrupted)
    if out.exists():raise CorpusError('Use a new continuation destination; preserve every attempt.')
    deadline=start+min(300,1800-protocol['preparation_seconds']-interrupted['wall_seconds'])
    base._check_deadline(deadline,'continuation_setup');out.mkdir(parents=True)
    for name in protocol['source_sha256']:
        shutil.copyfile(source/name,out/ALIASES.get(name,name))
    shutil.copyfile(source/'selected-configuration.json',out/'selected-configuration.json')
    shutil.copyfile(Path(PROTOCOL_PATH),out/'continuation-protocol.json')
    # Local read-only artifact reuse; no classifier or calibration is refitted.
    for folder in ('models','classical','primary-replay','replicas'):
        shutil.copytree(source/folder,out/folder,copy_function=os.link)
    marker=base._observation_marker(manifest['records_sha256']).with_name(
        manifest['records_sha256']+'-'+file_sha(source/'assessment-started.json')[:16]+'-continuation.json')
    write_json(marker,{'scope':base.REPRODUCTION_SCOPE,'registration_commit':registration_commit,'output':str(out),
                       'original_lock_sha256':file_sha(source/'assessment-started.json')})
    policies=read(out/'calibration-policy-selection.json');results=dict(partial['results']);saved=list(partial['predictions'])
    missing=[e for e in entries if e['key'] not in results]
    try:
        for entry in missing:
            key=entry['key'];base._snapshot(out/'continuation-progress.json',{'model':key,'wall_seconds':time.perf_counter()-start})
            print(json.dumps({'stage':'MISSING_FIXED_INFERENCE_ONLY','model':key}),flush=True)
            model=base._bounded_call(deadline,'continuation_load:'+key,base._load_entry,out,entry)
            try:result,predictions=evaluate_one(model,key,parts,policies[key],p,deadline)
            finally:del model;gc.collect()
            results[key]=result;saved+=predictions
        elapsed=time.perf_counter()-start
        report={'schema':'beto-v3-pilot-assessment','status':STATUS,'scope':base.REPRODUCTION_SCOPE,
            'data_manifest':manifest,'data_commit':selection['data_commit'],'protocol_commit':base.PROTOCOL_COMMIT,
            'selection_commit':protocol['original_selection_commit'],
            'selection_sha256':file_sha(out/'development-selection.json'),'policy_sha256':file_sha(out/'calibration-policy-selection.json'),
            'selected_model_key':selection['selected_model_key'],'best_development_candidate':selection['best_overall']['id'],
            'results':results,'replica_status':read(out/'replica-status.json'),'model_entries':entries,
            'prior_observation_evidence':base.known_observations(manifest['records_sha256']),
            'human_performance_measured':False,'new_assessment_is_now_observed':False,
            'cost':{'assessment_and_replica_wall_seconds':interrupted['wall_seconds']+elapsed,
                'original_phase_wall_seconds':interrupted['wall_seconds'],'original_phase_cap_seconds':900,
                'continuation_wall_seconds':elapsed,'continuation_cap_seconds':300,
                'active_phase_sum_excludes_development_pause':True,'continuation_process_memory':base.resource_report()},
            'continuation':{'registration_commit':registration_commit,'protocol_sha256':file_sha(out/'continuation-protocol.json'),
                'wall_seconds':elapsed,'completed_original_prefix':prefix,'filled_models':[e['key'] for e in missing],
                'new_training_calibration_or_selection':False},
            'limits':p['limits']+['Original 900 s phase interrupted; a disclosed inference-only amendment reused unused preparation allowance.',
                'Already observed reproduction; missing original raw results remain unavailable.',
                'Completed original results/probability prefix and locked policies are unchanged. Active phase sums exclude the implementation pause.']}
        validate_ledger(out,report,saved)
        base._bounded_call(deadline,'continued_predictions_write',write_json,out/'assessment-predictions.json',saved)
        base._bounded_call(deadline,'continued_report_write',write_json,out/'assessment.json',report)
        return report
    except BaseException as exc:
        if (out/'assessment.json').exists():(out/'assessment.json').rename(out/'assessment-interrupted-report.json')
        write_json(out/'continuation-interruption.json',{'error_type':type(exc).__name__,'reason':str(exc),
            'wall_seconds':time.perf_counter()-start,'complete':False,'observation_marker_retained':True})
        write_json(out/'continuation-partial-results.json',{'results':results,'predictions':saved,'complete':False})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--registration-commit',required=True)
    args=parser.parse_args();r=finish(args.source,args.output,args.registration_commit)
    print(json.dumps({'status':r['status'],'macro_f1':{k:v['classification']['macro_f1'] for k,v in r['results'].items()},'cost':r['cost']}))
