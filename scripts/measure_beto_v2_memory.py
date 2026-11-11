"""Measure inference backends from a light parent, outside the training process."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def measure(run):
    entries=json.loads((run/'evaluation-models.json').read_text(encoding='utf-8'))
    models={}
    for entry in entries:
        raw=subprocess.check_output([sys.executable,'-m','traveler_agent.beto_training_v2',
            'measure','--output',str(run),'--model-key',entry['key']],text=True)
        models[entry['key']]={**json.loads(raw),'launch_parent':'stdlib-only script; no numpy/torch/optimizer imports'}
    peaks=[value['peak_rss_mib'] for value in models.values()]
    valid=all(value is not None and value>0 for value in peaks) and len(set(peaks))>1
    report={'scope':'ISOLATED_INFERENCE_BACKENDS_NOT_TRAINING_MEMORY','models':models,
            'telemetry_comparison_valid':valid,
            'limits':['Includes imports, load and nine tune requests per process.',
                      'No assessment calls or threshold changes; training memory is separate.',
                      'Repeated identical high-water marks are rejected as uninformative telemetry.']}
    (run/'inference-memory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    result=measure(args.run)
    print(json.dumps({'telemetry_comparison_valid':result['telemetry_comparison_valid'],
                      'peak_rss_mib':{k:v['peak_rss_mib'] for k,v in result['models'].items()}}))
    if not result['telemetry_comparison_valid']:
        raise SystemExit('Memory comparison unavailable; do not report identical inherited peaks as backend measurements.')
