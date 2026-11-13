"""Real Gradio queue/session/upload/download acceptance, with explicit fallback."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traveler_agent.academic_acceptance import registered_cases, require_effective, record_violations, model_provenance, write_report, source_fingerprints


def run(commit, output, beto_artifact=None):
    if output.exists(): raise ValueError('Preserve the existing HTTP report.')
    _, fixtures, _ = registered_cases(commit)
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    if beto_artifact:
        os.environ['TRAVELER_BETO_ARTIFACT'] = str(beto_artifact.resolve())
    import httpx
    from gradio_client import Client, handle_file
    import app
    kinds = ('lr', 'svm', 'beto') if beto_artifact else ('lr', 'svm')
    report = {'scope': 'REAL_HTTP_QUEUE_SESSION_UPLOAD_DOWNLOAD_NOT_BROWSER_OR_REAL_VOICE',
              'registration_commit': commit, 'source_sha256': source_fingerprints(),
              'provenance': model_provenance(app, kinds), 'checks': [], 'models': {}, 'violations': []}

    def check(name, condition):
        report['checks'].append({'name': name, 'passed': bool(condition)})
        if not condition: report['violations'].append(name)

    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
    url = 'http://127.0.0.1:' + str(port)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='acceptance-http-', dir='artifacts') as temporary:
        work = Path(temporary).resolve()
        try:
            app.app.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1', server_port=port,
                share=False, inbrowser=False, prevent_thread_lock=True, quiet=True,
                **({'theme': app.THEME, 'css': app.UI_CSS} if app._GRADIO_MAJOR >= 6 else {}))
            with httpx.Client(trust_env=False, timeout=30) as connection:
                home = connection.get(url); config = connection.get(url + '/config')
            check('frontend_http200', home.status_code == 200)
            check('config_http200', config.status_code == 200)
            components = config.json()['components']
            check('lr_remains_default', any(c.get('type') == 'dropdown' and c.get('props', {}).get('label') == 'Modelo de demostración' and c['props'].get('value') == 'lr' for c in components))
            check('audio_microphone_and_upload_controls', any(c.get('type') == 'audio' and set(c.get('props', {}).get('sources', [])) == {'microphone', 'upload'} for c in components))
            client = Client(url, verbose=False, download_files=str(work / 'downloads'), httpx_kwargs={'trust_env': False})
            report['api_names'] = sorted(client.view_api(print_info=False, return_format='dict')['named_endpoints'])
            for kind in kinds:
                summary, encoded, view = client.predict(fixtures[5]['text'], kind, api_name='/analyze_with_agents')
                record = json.loads(encoded); require_effective(record, kind)
                check(kind + ':text_real_model_and_trace', not record_violations(record, fixtures[5]['text']) and 'TravelerOrchestrator' in view)
                _, invalid, invalid_view = client.predict('', kind, api_name='/analyze_with_agents')
                check(kind + ':empty_safe_error', invalid == '{}' and 'COMPLETADO' not in invalid_view)
                batch_result = client.predict(fixtures[0]['text'] + '\n' + fixtures[-1]['text'], kind, api_name='/run_batch_lines')
                check(kind + ':lines_status', 'analizado' in batch_result[-1])
                client.predict('demo-001', 'neutral', 'Verificación funcional; sin aprendizaje.', api_name='/revise')
                json_path, csv_path, _ = client.predict(api_name='/export_files')
                payload = json.loads(Path(json_path).read_text(encoding='utf-8'))
                records = payload['registros']
                exported = list(csv.DictReader(io.StringIO(Path(csv_path).read_text(encoding='utf-8-sig'))))
                check(kind + ':session_correction_download_json', len(records) == 2 and records[0]['etiqueta_corregida_humano'] == 'neutral' and records[0]['texto_original'] == fixtures[0]['text'])
                for row in records: require_effective(row, kind)
                check(kind + ':download_csv_provenance_and_formula', len(exported) == 2 and all(r['modelo_utilizado'] == kind for r in exported) and exported[-1]['texto_original'].startswith("'="))
                if kind == 'beto':
                    check('beto:download_csv_actual_manifest', all(r.get('modelo_solicitado') == 'beto' and r.get('fallback_modelo') == 'False' and r.get('version_modelo') == records[i]['seleccion_modelo']['manifest_sha256'] for i, r in enumerate(exported)))
                source = work / ('valid-' + kind + '.csv')
                source.write_text('reseña,nota\n"Una visita, con dos paradas\ny una pausa",fixture\n', encoding='utf-8-sig')
                uploaded = client.predict(handle_file(str(source)), kind, api_name='/run_batch_csv')
                check(kind + ':csv_upload_status', 'analizado' in uploaded[-1])
                json_path, _, _ = client.predict(api_name='/export_files')
                csv_records = json.loads(Path(json_path).read_text(encoding='utf-8'))['registros']
                check(kind + ':csv_upload_download_preserves_newline', len(csv_records) == 1 and csv_records[0]['texto_original'] == 'Una visita, con dos paradas\ny una pausa')
                for row in csv_records: require_effective(row, kind)
                source.write_text('reseña,reseña\nprimera,segunda\n', encoding='utf-8')
                bad = client.predict(handle_file(str(source)), kind, api_name='/run_batch_csv')
                check(kind + ':bad_csv_error', 'Error' in bad[-1])
                report['models'][kind] = {'single': record, 'downloaded_corrected_records': records, 'csv_export_rows': exported, 'uploaded_csv_records': csv_records}
            if beto_artifact:
                fault = work / 'bad-model'; fault.mkdir()
                (fault / 'model_manifest.json').write_text('{}', encoding='utf-8')
                os.environ['TRAVELER_BETO_ARTIFACT'] = str(fault)
                summary, encoded, _ = client.predict(fixtures[0]['text'], 'beto', api_name='/analyze_with_agents')
                record = json.loads(encoded); require_effective(record, 'beto', fallback_expected=True)
                check('fault_injection_real_lr_fallback_visible', 'respaldo' in summary and not record_violations(record, fixtures[0]['text']))
                client.predict(fixtures[1]['text'], 'beto', api_name='/run_batch_lines')
                json_path, csv_path, _ = client.predict(api_name='/export_files')
                fallback_records = json.loads(Path(json_path).read_text(encoding='utf-8'))['registros']
                for row in fallback_records: require_effective(row, 'beto', fallback_expected=True)
                exported = list(csv.DictReader(io.StringIO(Path(csv_path).read_text(encoding='utf-8-sig'))))
                check('fault_injection_export_provenance', len(exported) == 1 and exported[0]['modelo_solicitado'] == 'beto' and exported[0]['modelo_utilizado'] == 'lr' and exported[0]['fallback_modelo'] == 'True')
                report['fallback'] = {'scope': 'INVALID_ARTIFACT_REAL_LR_NOT_REAL_BETO', 'single': record, 'exported_records': fallback_records}
                os.environ['TRAVELER_BETO_ARTIFACT'] = str(beto_artifact.resolve())
            report['status'] = 'PASSED' if not report['violations'] else 'FAILED'
        except Exception as exc:
            report.update(status='FAILED', failure_type=type(exc).__name__)
            raise
        finally:
            app.app.close()
            report['wall_seconds'] = time.perf_counter() - start
            write_report(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--beto-artifact', type=Path)
    args = parser.parse_args()
    report = run(args.registration_commit, args.output, args.beto_artifact)
    print(json.dumps({'status': report['status'], 'checks': len(report['checks']), 'violations': report['violations']}))
    if report['status'] != 'PASSED': raise SystemExit(1)
