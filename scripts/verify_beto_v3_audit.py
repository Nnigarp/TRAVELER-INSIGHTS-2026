"""Recompute the frozen quality report, without sentiment assessment outputs."""
import argparse,json
from pathlib import Path
from traveler_agent.beto_training_v3_data import automatic_audit,load_data
from traveler_agent.corpus import CorpusError
from traveler_agent.synthetic_diversity import canonical_sha,ignored_target,write_json


def compare(actual,reference):
    if canonical_sha(actual)!=canonical_sha(reference):
        raise CorpusError('Recomputed corpus audit differs from the frozen reference; investigate without changing data or gates.')
    return {'status':'EXACT_FROZEN_AUDIT_REPRODUCED','automatic_gates_pass':actual['automatic_gates_pass'],
        'records_sha256':actual['manifest']['records_sha256'],
        'shape_only_macro_f1':actual['shortcuts']['length_and_punctuation']['macro_f1'],
        'model_or_data_decisions_changed':False,'assessment_predictions_used':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    reference=json.loads(Path('docs/research/beto-v3-quality.json').read_text(encoding='utf-8'))
    load_data(require_review=True)
    result=compare(automatic_audit(a.cache),reference)
    write_json(ignored_target(a.out,kinds=('artifacts',)),result);print(json.dumps(result))
