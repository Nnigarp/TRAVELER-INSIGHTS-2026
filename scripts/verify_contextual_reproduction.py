"""Compare fixed predictions across software reproductions, never tune a model."""
import argparse
import json
from pathlib import Path

import numpy as np


def verify(actual, reference):
    def indexed(rows):
        keys=[(r['model'],r['review_id']) for r in rows]
        if len(set(keys))!=len(keys):raise ValueError('Duplicate prediction key')
        return dict(zip(keys,rows))
    left,right=indexed(actual),indexed(reference)
    if set(left)!=set(right):raise ValueError('Prediction membership differs')
    maximum=0.
    for key, row in left.items():
        other=right[key]
        for field in ('family','author_intent','prediction','review'):
            if row[field]!=other[field]:raise ValueError('Fixed outcome differs')
        a,b=np.asarray(row['probabilities']),np.asarray(other['probabilities'])
        if a.shape!=(3,) or b.shape!=(3,) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Invalid probability vector')
        maximum=max(maximum,float(np.max(np.abs(a-b))))
    if maximum>1e-5:raise ValueError('Probability drift exceeds numerical reproduction tolerance')
    return {'fixed_predictions':len(left),'maximum_probability_difference':maximum,
            'status':'REPRODUCED_FIXED_SYNTHETIC_OUTCOMES_NOT_NEW_TEST_EVIDENCE'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(verify(json.loads(args.actual.read_text()),json.loads(args.reference.read_text()))))


if __name__=='__main__':main()
