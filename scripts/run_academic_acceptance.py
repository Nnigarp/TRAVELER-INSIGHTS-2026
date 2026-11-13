"""Run registered functional contracts or a separately locked synthetic measure."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traveler_agent.academic_acceptance import run_functional, run_predictive, write_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('functional', 'predictive'))
    parser.add_argument('--registration-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--beto-artifact', type=Path)
    parser.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a new output, preserving prior evidence.')
    if args.beto_artifact:
        os.environ['TRAVELER_BETO_ARTIFACT'] = str(args.beto_artifact.resolve())
    if args.mode == 'functional':
        kinds = ('lr', 'svm', 'beto') if args.beto_artifact else ('lr', 'svm')
        report = run_functional(args.registration_commit, kinds)
        write_report(args.output, report)
        print(json.dumps({'status': report['status'], 'checks': len(report['checks']), 'violations': report['violations'], 'models': list(report['models'])}))
        if report['status'] != 'PASSED': raise SystemExit(1)
    else:
        report = run_predictive(args.registration_commit, args.output, reproduce=args.reproduce)
        print(json.dumps({'status': report['status'], 'scope': report['scope'], 'models': {k: {'macro_f1': v['metrics']['macro_f1'], 'negative_to_positive': v['metrics']['negative_to_positive_count'], 'automatic_negative_to_positive': v['metrics']['abstention']['automatic_negative_to_positive_count'], 'ambiguity': v['ambiguity']} for k, v in report['models'].items()}}))
