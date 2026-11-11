"""Refuse configuration drift from the development choices committed before assessment."""
import argparse
import json
from pathlib import Path


def choices(selection):
    return {'records_sha256':selection['records_sha256'],
            'protocol_sha256':selection['protocol_sha256'],
            'registration_commit':selection['registration_commit'],
            'primary_seed':selection['primary_seed'],
            'classical':{key:value['selected']['C'] for key,value in selection['classical_selection'].items()},
            'heads':{key:value['selected']['epoch'] for key,value in selection['head_selection'].items()},
            'grid':[{key:row.get(key) for key in ('id','status','configuration','head_epoch','selected_epoch')}
                    for row in selection['candidates']],
            'candidate':{key:selection['selected_candidate'][key]
                         for key in ('id','configuration','head_epoch','selected_epoch')},
            'selected_model_key':selection['selected_model_key']}


def verify(actual,reference):
    if choices(actual)!=choices(reference):
        raise ValueError('Development selection drifted from pre-assessment lock; stop, do not retune on assessment.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args()
    verify(json.loads(args.actual.read_text(encoding='utf-8')),json.loads(args.reference.read_text(encoding='utf-8')))
    print('All eight executed candidates, selected C values, head epochs, configuration and seed match the pre-assessment lock.')
