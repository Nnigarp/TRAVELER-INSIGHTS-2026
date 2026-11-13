"""Bounded local metadata only: no training, model load, reviews or network."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traveler_agent.beto_distribution import artifact_available
from traveler_agent.corpus import file_sha


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a new diagnostic file.')
    root = args.root.resolve()
    result = {'python': sys.version.split()[0], 'platform': platform.platform(),
        'cpu_logical': os.cpu_count(), 'scope': 'LOCAL_METADATA_NO_TRAINING_OR_INFERENCE',
        'registered_replay_cap_seconds': 900, 'models': [], 'interruptions': [], 'dependencies': {}}
    for name in ('torch', 'transformers', 'safetensors', 'gradio'):
        try: result['dependencies'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result['dependencies'][name] = 'NOT_INSTALLED'
    try:
        import psutil
        memory = psutil.virtual_memory()
        result['ram_total_mib'] = round(memory.total / 1048576)
        result['ram_available_mib_at_check'] = round(memory.available / 1048576)
    except ImportError: result['ram_available_mib_at_check'] = None
    if root.is_dir():
        for path in list(root.rglob('model_manifest.json'))[:100]:
            if path.stat().st_size > 65536: continue
            result['models'].append({'relative_directory': str(path.parent.relative_to(root)),
                'manifest_sha256': file_sha(path), 'complete_file_presence': artifact_available(path.parent)})
        for path in list(root.rglob('*interruption.json'))[:100]:
            if path.stat().st_size > 65536: continue
            try: data = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError): continue
            allowed = ('status', 'stage', 'wall_seconds', 'wall_seconds_cap', 'effective_wall_seconds_cap',
                       'artifact_complete_within_budget', 'assessment_used', 'selection_retuned')
            result['interruptions'].append({'relative_file': str(path.relative_to(root)),
                **{k: data[k] for k in allowed if k in data and isinstance(data[k], (str, int, float, bool))}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2); stream.write('\n')
    print(json.dumps({'models': len(result['models']), 'interruptions': len(result['interruptions']),
                      'training_or_inference_executed': False}))
