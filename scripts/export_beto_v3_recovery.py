"""Export measured observed reproduction, never manufacture original raw results.

Writes local research files only. It does not push to GitHub or publish weights.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from traveler_agent import beto_training_v3 as base
from traveler_agent.beto_training_v3_analysis import analyze
from traveler_agent.beto_training_v3_recovery import SELECTION_COMMIT
from traveler_agent.corpus import CorpusError,file_sha
from traveler_agent.synthetic_diversity import ignored_target,write_json


def encode(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()


def write_same_or_new(path,payload):
    if path.exists():
        if path.read_bytes()!=payload:raise CorpusError('Preserve existing evidence: '+str(path))
        return
    with path.open('xb') as f:f.write(payload)


def require_observed_reproduction(report,bindings):
    if (report.get('status')!='COMPLETED_WITHIN_REGISTERED_BUDGET'
            or report.get('scope')!=base.REPRODUCTION_SCOPE
            or report.get('human_performance_measured') is not False
            or report.get('selection_commit')!=SELECTION_COMMIT
            or not report.get('prior_observation_evidence')
            or bindings.get('recovery',{}).get('original_raw_results_available') is not False
            or report.get('data_manifest',{}).get('records_sha256')!=bindings.get('records_sha256')):
        raise CorpusError('Only complete measured observed reproduction can replace the missing-reference blocker.')


def provenance(root,report,bindings):
    require_observed_reproduction(report,bindings)
    return {'schema':'beto-v3-measured-reference-provenance',
        'source_scope':'MEASURED_FIXED_REPRODUCTION_NOT_ORIGINAL_RAW_RECOVERY',
        'already_observed':True,'human_gold':False,'original_raw_probabilities_available':False,
        'records_sha256':bindings['records_sha256'],'data_commit':bindings['data_commit'],
        'original_selection_commit':SELECTION_COMMIT,
        'recovery_registration_commit':bindings['recovery']['protocol_registration_commit'],
        'assessment_report_sha256':file_sha(root/'assessment.json'),
        'predictions_sha256':file_sha(root/'assessment-predictions.json'),
        'limits':['New probabilities came from actual fixed-recipe inference; missing old probabilities were not reconstructed.',
                  'CI compares software replay with this observed reproduction, not with unavailable original raw files.',
                  'No new final test, model/epoch selection or human validation.']}


def export(root):
    root=ignored_target(root,kinds=('artifacts',));dest=Path('docs/research')
    read=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    report=read('assessment.json');bindings=read('development-selection.json')
    require_observed_reproduction(report,bindings)
    base.verify_selection_commit(bindings,SELECTION_COMMIT,reproduce_observed=True)
    analysis=analyze(root);ref=provenance(root,report,bindings)
    old=json.loads((dest/'beto-v3-prior-reported-aggregates.json').read_text(encoding='utf-8'))
    comparisons={}
    for key,value in old['reported_macro_f1'].items():
        measured=report['results'].get(key,{}).get('classification',{}).get('macro_f1')
        comparisons[key]={'prior_reported_macro_f1':value,'new_measured_macro_f1':measured,
            'status':'OBSERVED' if measured is not None else 'NOT_MEASURED_NO_IMPUTATION',
            'difference':measured-value if measured is not None else None}
    comparison={'scope':'NEW_MEASUREMENTS_VERSUS_PRIOR_REPORTED_AGGREGATES_NOT_RAW_EQUALITY',
        'metrics':comparisons,'original_raw_probability_equality_verifiable':False,
        'model_configuration_changed_to_match':False}
    measured=bindings['recovery']['cost']['wall_seconds']+report['cost']['assessment_and_replica_wall_seconds']
    if measured>1800:raise CorpusError('Reconstruction exceeded the registered combined compute budget.')
    execution={'scope':'MEASURED_FIXED_OBSERVED_RECONSTRUCTION',
        'recovery_registration_commit':bindings['recovery']['protocol_registration_commit'],
        'fixed_preparation_cost':bindings['recovery']['cost'],'assessment_and_replica_cost':report['cost'],
        'measured_combined_wall_seconds':measured,'combined_cap_seconds':1800,
        'original_development_cost_archived_not_in_this_total':bindings['cost'],
        'original_development_cost_source_commit':SELECTION_COMMIT,'human_performance_measured':False,
        'replica_status':report['replica_status'],
        'initial_900_plus_450_contingency_activated':False,'monetary_cost_measured':False}
    mapping={'assessment.json':'beto-v3-assessment.json',
        'assessment-predictions.json':'beto-v3-predictions.json',
        'calibration-policy-selection.json':'beto-v3-calibration-policy.json',
        'development-selection.json':'beto-v3-recovered-bindings.json',
        'replica-status.json':'beto-v3-replica-status.json'}
    for row in report['replica_status']:
        if row['status']=='EXECUTED':
            mapping[f"replicas/{row['seed']}/replay.json"]=f"beto-v3-seed{row['seed']}-replay.json"
    # Copies preserve actual source bytes. No prediction is derived from old scores.
    for source,target in mapping.items():write_same_or_new(dest/target,(root/source).read_bytes())
    for target,value in [('beto-v3-analysis.json',analysis),('beto-v3-reference-provenance.json',ref),
                         ('beto-v3-aggregate-reproduction-comparison.json',comparison),
                         ('beto-v3-recovery-execution.json',execution)]:write_same_or_new(dest/target,encode(value))
    write_same_or_new(root/'analysis.json',encode(analysis))
    return {'exported_measured_models':len(report['results']),'rows':analysis['assessment_rows'],
            'scope':ref['source_scope'],'current_combined_wall_seconds':measured}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(export(args.run)))
