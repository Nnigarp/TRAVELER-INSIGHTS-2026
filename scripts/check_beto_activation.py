"""Verify actual supervised V3 execution, never credit a classical fallback."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Preserve prior activation evidence; use a new output.')
    os.environ['TRAVELER_BETO_ARTIFACT'] = str(args.artifact.resolve())
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    from traveler_agent.beto_distribution import MODEL_MANIFEST_SHA, artifact_available
    from traveler_agent.academic_acceptance import require_effective, record_violations, write_report
    from traveler_agent.corpus import file_sha
    if not artifact_available(args.artifact): parser.error('Complete trained artifact unavailable; no LR substitute is accepted.')
    if file_sha(args.artifact / 'model_manifest.json') != MODEL_MANIFEST_SHA: parser.error('This activation checks the recovered PR28 V3 release.')
    import app
    text = 'La explicación se entendía muy bien; al final, sin embargo, nos dejaron sin transporte de regreso.'
    summary, encoded, view = app.analyze_with_agents(text, 'beto')
    record = json.loads(encoded)
    require_effective(record, 'beto')
    problems = record_violations(record, text)
    if 'beto' not in [v for _, v in app._MODEL_CHOICES]: problems.append('beto_not_in_ui_choices')
    if 'TravelerOrchestrator' not in view: problems.append('agent_view_missing')
    defaults = [c.get('props', {}).get('value') for c in app.app.config['components']
                if c.get('type') == 'dropdown' and c.get('props', {}).get('label') == 'Modelo de demostración']
    if defaults != ['lr']: problems.append('default_model_changed')
    report = {'scope': 'ACTUAL_RECOVERED_ARTIFACT_INFERENCE_NOT_NEW_PREDICTIVE_EVALUATION',
        'status': 'PASSED' if not problems else 'FAILED', 'modelo_utilizado': record['modelo_utilizado'],
        'fallback': record['seleccion_modelo']['fallback'], 'model_manifest_sha256': MODEL_MANIFEST_SHA,
        'trace_events': len(record['multiagente']['traza']), 'ui_beto_available': app._BETO_AVAILABLE,
        'default_model': defaults[0] if defaults else None, 'violations': problems, 'observed': record}
    write_report(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'observed'}, ensure_ascii=False))
    if report['status'] != 'PASSED': raise SystemExit(1)
