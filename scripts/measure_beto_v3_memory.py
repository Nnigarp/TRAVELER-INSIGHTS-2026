"""Measure each V3 backend from a stdlib-only parent, never the optimizer parent."""
import argparse,json,subprocess,sys
from pathlib import Path


def measure(run):
    entries=json.loads((run/'evaluation-models.json').read_text(encoding='utf-8'));models={}
    for entry in entries:
        raw=subprocess.check_output([sys.executable,'-m','traveler_agent.beto_training_v3','measure',
              '--output',str(run),'--model-key',entry['key']],text=True)
        models[entry['key']]={**json.loads(raw),'launch_parent':'stdlib-only; no numpy/torch/optimizer imports'}
    peaks=[value['peak_rss_mib'] for value in models.values()]
    valid=bool(peaks) and all(v is not None and v>0 for v in peaks) and len(set(peaks))>1
    result={'scope':'ISOLATED_INFERENCE_NOT_TRAINING_PEAK','models':models,'telemetry_comparison_valid':valid,
            'limits':['Process imports, admission verification, artifact load and nine tune requests; no assessment calls.',
                      'Host-specific RSS peaks, not guaranteed RAM requirements on another platform.']}
    (run/'inference-memory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    r=measure(args.run);print(json.dumps({'valid':r['telemetry_comparison_valid'],'peak_rss_mib':{k:v['peak_rss_mib'] for k,v in r['models'].items()}}))
    if not r['telemetry_comparison_valid']:raise SystemExit('Unavailable or inherited memory peaks; do not compare backends.')
