from copy import deepcopy
import pytest
from scripts.verify_beto_v3_audit import compare
from traveler_agent.corpus import CorpusError


def test_quality_reproduction_preserves_numeric_evidence_not_just_pass_flags():
    r={'automatic_gates_pass':True,'manifest':{'records_sha256':'fixed'},
       'shortcuts':{'length_and_punctuation':{'macro_f1':.385}},'duplicates':{'candidates':[]}}
    assert compare(deepcopy(r),r)['assessment_predictions_used'] is False
    changed=deepcopy(r);changed['shortcuts']['length_and_punctuation']['macro_f1']=.4
    with pytest.raises(CorpusError,match='differs'):compare(changed,r)


def test_different_corpus_cannot_pass_by_retaining_a_pass_flag():
    with pytest.raises(CorpusError,match='differs'):
        compare({'automatic_gates_pass':True,'manifest':{'records_sha256':'other'}},
                {'automatic_gates_pass':True,'manifest':{'records_sha256':'fixed'}})
