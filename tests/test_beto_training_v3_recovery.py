from copy import deepcopy
import json
from pathlib import Path
import pytest
from traveler_agent import beto_training_v3_recovery as recovery
from traveler_agent.corpus import CorpusError


def archived():return json.loads(Path('docs/research/beto-v3-selection.json').read_text())


def test_recovery_keeps_exact_archived_recipes_without_final_score_input():
    original=archived();before=deepcopy(original)
    assert set(recovery.fixed_recipes(original))==recovery.EXPECTED_KEYS
    assert original==before


@pytest.mark.parametrize('change',['epoch','model','corpus_commit'])
def test_changed_recipe_is_not_a_recovery(change):
    s=archived()
    if change=='epoch':s['selected_adjusted']['selected_epoch']=2
    elif change=='model':s['model_entries'].append({'key':'unregistered'})
    else:s['data_commit']='0'*40
    with pytest.raises(CorpusError):recovery.fixed_recipes(s)


def test_classical_recovery_uses_one_archived_c_and_training_only(monkeypatch):
    train=[{'role':'train'}];s=archived()
    def fit(rows,config):
        assert rows is train and config is s['classical_selection']
        return {'fixed':'models'}
    monkeypatch.setattr(recovery,'fit_represented_baselines',fit)
    assert recovery.fit_fixed_classics(train,s)=={'fixed':'models'}
    with pytest.raises(CorpusError,match='TRAIN'):
        recovery.fit_fixed_classics([{'role':'assessment'}],s)


def test_observation_reservation_survives_another_output_directory(tmp_path,monkeypatch):
    marker=tmp_path/'observed/fixed.json'
    monkeypatch.setattr(recovery.base,'_observation_marker',lambda _:marker)
    recovery.reserve_observed({'records_sha256':'fixed'})
    first=marker.read_bytes();recovery.reserve_observed({'records_sha256':'fixed'})
    assert marker.read_bytes()==first and json.loads(first)['new_test'] is False


def test_registered_fingerprint_rejects_a_different_learning_rate(monkeypatch):
    s=archived();p=json.loads(Path(recovery.RECOVERY_PATH).read_text())
    monkeypatch.setattr(recovery.base,'_git_json',lambda *_:p)
    manifest={'records_sha256':s['records_sha256']}
    recovery.verify_protocol('unused',manifest,s)
    changed=deepcopy(s);changed['selected_adjusted']['configuration']['encoder_lr']=.01
    with pytest.raises(CorpusError):recovery.verify_protocol('unused',manifest,changed)
