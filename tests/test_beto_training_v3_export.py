from copy import deepcopy
import pytest
from scripts.export_beto_v3_recovery import write_same_or_new,require_observed_reproduction
from traveler_agent.beto_training_v3 import REPRODUCTION_SCOPE
from traveler_agent.beto_training_v3_recovery import SELECTION_COMMIT
from traveler_agent.corpus import CorpusError


def test_existing_measurement_is_immutable_and_repeatable(tmp_path):
    p=tmp_path/'evidence.json';write_same_or_new(p,b'{"measured":1}')
    write_same_or_new(p,b'{"measured":1}')
    with pytest.raises(CorpusError,match='Preserve'):write_same_or_new(p,b'{"measured":2}')
    assert p.read_bytes()==b'{"measured":1}'


def fixture():
    report={'status':'COMPLETED_WITHIN_REGISTERED_BUDGET','scope':REPRODUCTION_SCOPE,
            'human_performance_measured':False,'selection_commit':SELECTION_COMMIT,
            'prior_observation_evidence':[{'source':'prior'}],'data_manifest':{'records_sha256':'fixed'}}
    binding={'records_sha256':'fixed','recovery':{'original_raw_results_available':False}}
    return report,binding


@pytest.mark.parametrize('mutation',['fresh','unmeasured','no_prior','other_data'])
def test_old_aggregates_or_new_test_claim_cannot_be_exported_as_recovered_raw(mutation):
    report,binding=fixture();require_observed_reproduction(report,binding)
    report=deepcopy(report)
    if mutation=='fresh':report['scope']='FRESH_SYNTHETIC_PILOT_ASSESSMENT'
    elif mutation=='unmeasured':report['status']='REPORTED_BY_PRIOR_CHAT_ONLY'
    elif mutation=='no_prior':report['prior_observation_evidence']=[]
    else:binding['records_sha256']='other'
    with pytest.raises(CorpusError,match='observed reproduction'):require_observed_reproduction(report,binding)
