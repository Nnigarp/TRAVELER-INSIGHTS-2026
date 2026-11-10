"""Reproduction tolerance fixed before assessment; no scientific model selection."""
import argparse
import json
from pathlib import Path

import numpy as np


def verify(actual,reference):
    def index(values):
        keys=[(r['model'],r['review_id']) for r in values]
        if len(set(keys))!=len(keys):raise ValueError('Duplicate model/review prediction')
        return dict(zip(keys,values))
    left,right=index(actual),index(reference)
    if set(left)!=set(right):raise ValueError('Model/review membership changed')
    raw_delta=0.;cal_delta=0.
    for key,row in left.items():
        other=right[key]
        for field in ('family','intended_label','prediction'):
            if row[field]!=other[field]:raise ValueError('Fixed synthetic class outcome changed')
        for field in ('probabilities','calibrated_probabilities'):
            a,b=np.asarray(row[field]),np.asarray(other[field])
            if a.shape!=(3,) or b.shape!=(3,) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid probabilities')
            delta=float(np.max(np.abs(a-b)))
            if field=='probabilities':raw_delta=max(raw_delta,delta)
            else:cal_delta=max(cal_delta,delta)
    if raw_delta>.003 or cal_delta>.005:raise ValueError('Numerical training reproduction tolerance exceeded')
    return {'status':'FIXED_SYNTHETIC_OUTCOMES_REPRODUCED_NOT_NEW_TEST','predictions':len(left),
            'max_raw_probability_difference':raw_delta,'max_calibrated_difference':cal_delta,
            'tolerance_preassessment':{'raw':.003,'calibrated':.005}}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--actual',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(verify(json.loads(args.actual.read_text()),json.loads(args.reference.read_text()))))


if __name__=='__main__':main()
