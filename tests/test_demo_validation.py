"""Check functional contracts; never assert synthetic examples are human gold."""
from copy import deepcopy

from traveler_agent.demo_validation import run_validation,trace_violations


def test_functional_report_executes_five_agents_batch_and_export_without_accuracy():
    report=run_validation(model_kinds=('lr',))
    assert report['contract_status']=='PASSED' and not report['violations']
    model=report['models']['lr']
    assert len(model['cases'])==model['batch_rows']==6
    assert model['exports']['json_roundtrip_rows']==model['exports']['csv_roundtrip_rows']==6
    assert model['audio']['status'].startswith('PENDING')
    assert report['human_gold_reviews']==0
    assert 'accuracy' not in model
    record=model['cases'][0]['observed']
    assert trace_violations(record)==[]
    broken=deepcopy(record);broken['multiagente']['traza']=broken['multiagente']['traza'][:-1]
    assert 'trace_sequence_invalid' in trace_violations(broken)


def test_functional_ui_callbacks_run_without_browser_or_real_voice():
    report=run_validation(model_kinds=('lr',),include_ui=True)
    assert report['contract_status']=='PASSED'
    assert report['models']['lr']['ui']['status']=='PASSED_CALLBACKS_BROWSER_NOT_VALIDATED'
