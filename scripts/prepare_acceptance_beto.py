"""One fixed TRAIN-only reconstruction when no trained V3 artifact is available."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traveler_agent.academic_acceptance import registered_cases
from traveler_agent import beto_training_v3 as runner


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration-commit', required=True)
    parser.add_argument('--cache', type=Path, default=Path('artifacts/beto-v3-checkpoint'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    registered_cases(args.registration_commit)
    selection = Path('docs/research/beto-v3-selected-configuration.json')
    recipe = json.loads(selection.read_text(encoding='utf-8'))
    registration = args.output.parent / (args.output.name + '-registration')
    runner.register(registration, recipe['data_commit'])
    receipt = runner.replay(args.output, registration, args.cache, recipe['data_commit'], selection, 42, download=args.download)
    print(json.dumps({'artifact': receipt['artifact'], 'assessment_used': receipt['assessment_used'],
                      'selection_retuned': receipt['selection_retuned'], 'cost': receipt['cost']}))
