"""Registered acceptance of actual application options, without model selection.

Functional contracts and synthetic intended-label measurements are separate.
No downloads, training, UI startup or predictions occur on import.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import importlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from types import SimpleNamespace
import wave

import numpy as np

from .corpus import CorpusError, file_sha, text_sha
from .demo_data import DEMO
from .demo_validation import trace_violations
from .labels import LABELS
from .research_metrics import summarize_predictions

DATA = Path(__file__).parent / 'research_data'
REGISTERED = tuple(DATA / ('academic_acceptance_' + name + '.json')
                   for name in ('protocol', 'functional', 'predictive'))
PUBLISHED_PREDICTIVE = Path('docs/acceptance/2026-11-13-predictive.json')


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def registered_cases(commit):
    """Check actual committed bytes, never a caller-supplied admitted flag."""
    if not re.fullmatch(r'[0-9a-f]{40}', commit or ''):
        raise CorpusError('A complete pre-execution registration commit is required.')
    for path in REGISTERED:
        relative = 'src/traveler_agent/research_data/' + path.name
        old = subprocess.run(['git', 'show', commit + ':' + relative], check=True,
                             capture_output=True, timeout=30).stdout
        if old != path.read_bytes():
            raise CorpusError('Registered cases or criteria changed: ' + path.name)
    protocol, functional, predictive = [json.loads(p.read_text(encoding='utf-8')) for p in REGISTERED]
    all_rows = functional + predictive
    if len({r['id'] for r in all_rows}) != len(all_rows):
        raise CorpusError('Duplicate acceptance identifiers.')
    if len({text_sha(r['text']) for r in all_rows}) != len(all_rows):
        raise CorpusError('Functional and predictive texts must be distinct.')
    if Counter(r['intended_label'] for r in predictive) != Counter({'negativo': 12, 'neutral': 12, 'positivo': 12, None: 8}):
        raise CorpusError('Registered predictive membership changed.')
    if {text_sha(r['text']) for r in predictive} & {text_sha(t) for t, _ in DEMO}:
        raise CorpusError('An acceptance case duplicates DEMO training.')
    return protocol, functional, predictive


def require_effective(record, requested, *, fallback_expected=False):
    """A BETO requirement cannot pass with a perfectly working LR fallback."""
    selection = record.get('seleccion_modelo', {})
    expected = 'lr' if fallback_expected else requested
    if record.get('modelo_utilizado') != expected:
        raise CorpusError('Requested model did not execute: ' + requested)
    if requested == 'beto' and (selection.get('requested') != 'beto'
            or selection.get('effective') != expected
            or selection.get('fallback') is not fallback_expected):
        raise CorpusError('BETO provenance/fallback contract violated.')


def record_violations(record, text):
    problems = trace_violations(record)
    meta = record.get('multiagente', {})
    if text in json.dumps(meta.get('traza', []), ensure_ascii=False):
        problems.append('raw_text_in_metadata_trace')
    if any(span not in text for span in record.get('fragmentos_literales_no_explicativos', [])):
        problems.append('nonliteral_evidence')
    scores = record.get('puntuaciones_modelo_no_calibradas', {})
    if set(scores) != set(LABELS) or not np.isfinite(list(scores.values())).all() or any(v < 0 for v in scores.values()) or abs(sum(scores.values()) - 1) > .0003:
        problems.append('invalid_scores')
    # Returned confidence is rounded to four decimals. Check the actual legacy
    # rule only where rounding cannot straddle its strict boundary.
    confidence = record.get('confianza_no_calibrada', -1)
    contrast = bool(re.search(r'\b(pero|aunque|sin embargo)\b', text, re.I))
    expected = []
    uncertain = abs(confidence - .56) <= .000051 or (contrast and abs(confidence - .8) <= .000051)
    if confidence < .56:
        expected.append('SCORE_DEMO_BAJO')
    if contrast and confidence < .8:
        expected.append('CONTRASTE_CON_SCORE_LIMITADO')
    if not uncertain and (meta.get('razones_revision') != expected or record.get('revision_humana') != bool(expected)):
        problems.append('review_rule_or_reasons_inconsistent')
    return problems


def model_provenance(app, kinds):
    from . import model_runtime as runtime
    result = {}
    for kind in kinds:
        if kind != 'beto':
            app.get_demo_model(kind)
            result[kind] = {'training_source': 'HISTORICAL_SYNTHETIC_DEMO', 'training_rows': len(DEMO),
                            'training_classes': dict(Counter(y for _, y in DEMO)),
                            'demo_source_sha256': file_sha(Path(__file__).with_name('demo_data.py'))}
            if kind == 'lr':
                result[kind]['actual_runtime_artifact_sha256'] = file_sha(app.MODEL_PATH)
        else:
            path = runtime.configured_artifact()
            if path is None:
                raise CorpusError('BETO V3 required; no configured trained artifact.')
            digest = file_sha(path / 'model_manifest.json')
            model = runtime._load(str(path), digest)
            training = model.metadata.get('training', {})
            if (training.get('experiment') != 'BETO_V3_QUALITY_PILOT' or training.get('seed') != 42
                    or model.metadata.get('mode') != 'full' or model.encoder.max_length != 256
                    or training.get('records_sha256') != '1ca1521e303325bb314eb6d8c2d8e8b42e1510ac5352a541ab119bd403c16f9d'
                    or training.get('head_epoch') != 80 or training.get('encoder_epoch') != 1):
                raise CorpusError('The BETO acceptance requirement is specifically fixed V3 seed42.')
            result[kind] = {'training_source': 'EXPERIMENTAL_SYNTHETIC_V3', 'manifest_sha256': digest,
                            'training': training, 'max_length': model.encoder.max_length,
                            'checkpoint_revision': model.metadata['revision']}
    return result


def source_fingerprints():
    root = Path(__file__).resolve().parents[2]
    return {str(p.relative_to(root)): file_sha(p) for p in
            (Path(__file__), root / 'app.py', Path(__file__).with_name('batch.py'),
             Path(__file__).with_name('model_runtime.py'), Path(__file__).with_name('multiagent.py'))}


def novelty_audit(commit):
    """Text-only overlap description; no sentiment model or policy is fitted."""
    _, functional, rows = registered_cases(commit)
    from .beto_training_v3_data import load_data
    from sklearn.feature_extraction.text import TfidfVectorizer
    parts, _, _ = load_data(require_review=True)
    prior = [t for t, _ in DEMO] + [r['text'] for r in parts['train']]
    targets = [r['text'] for r in rows]
    if {text_sha(t) for t in prior} & {text_sha(t) for t in targets}:
        raise CorpusError('Exact overlap with DEMO or V3 TRAIN; do not evaluate.')
    # This vocabulary is used only by the audit and is never passed to a model.
    vectors = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5)).fit_transform(prior + [r['text'] for r in functional] + targets)
    reference = vectors[:-len(rows)]; target = vectors[-len(rows):]
    similarities = (target @ reference.T).toarray()
    within = (target @ target.T).toarray(); np.fill_diagonal(within, 0)
    return {'scope': 'DESCRIPTIVE_TEXT_OVERLAP_NOT_PROOF_OF_SEMANTIC_INDEPENDENCE',
            'registered_commit': commit, 'exact_training_overlap': 0,
            'compared_training_rows': len(prior), 'internal_max_character_cosine': float(within.max()),
            'nearest_training_or_functional_character_cosine': {row['id']: float(value) for row, value in zip(rows, similarities.max(axis=1))},
            'classifier_features_or_labels_used': False, 'cases_edited_using_model_predictions': False}


def token_diagnostic(app, kind, text):
    if kind != 'beto':
        return None
    from . import model_runtime as runtime
    path = runtime.configured_artifact()
    model = runtime._load(str(path), file_sha(path / 'model_manifest.json'))
    ids = model.encoder.tokenizer(text, truncation=False, verbose=False)['input_ids']
    return {'input_tokens_including_special': len(ids), 'window_tokens': model.encoder.max_length,
            'truncated': len(ids) > model.encoder.max_length,
            'scope': 'TOKENIZATION_DIAGNOSTIC_NOT_AN_EXPLANATION_OF_PREDICTION'}


def observe(app, kind, row):
    start = time.perf_counter()
    summary, encoded, view = app.analyze_with_agents(row['text'], kind)
    record = json.loads(encoded)
    require_effective(record, kind)
    problems = record_violations(record, row['text'])
    if not summary or 'TravelerOrchestrator' not in view:
        problems.append('summary_or_agent_view_missing')
    if kind == 'beto' and ('ventana' not in summary or '*Modelo DEMO entrenado' in summary):
        problems.append('beto_source_or_token_limit_disclosure_missing')
    return {'id': row['id'], 'text': row['text'], 'record': record,
            'latency_ms': (time.perf_counter() - start) * 1000,
            'tokenization': token_diagnostic(app, kind, row['text']), 'violations': problems}


def run_functional(commit, kinds=('lr', 'svm')):
    protocol, fixtures, _ = registered_cases(commit)
    app = importlib.import_module('app')
    report = {'schema': 'academic-functional-v1', 'scope': 'FUNCTIONAL_NOT_PREDICTIVE_ACCURACY',
              'registration_commit': commit, 'provenance': model_provenance(app, kinds),
              'models': {}, 'checks': [], 'violations': [], 'human_gold': False,
              'source_sha256': source_fingerprints()}

    def check(name, value):
        report['checks'].append({'name': name, 'passed': bool(value)})
        if not value:
            report['violations'].append(name)

    from .audio_asr import recognize_audio, dependency_versions
    class StubASR:
        def transcribe(self, *args, **kwargs):
            return iter([SimpleNamespace(text='El recorrido fue malo.')]), SimpleNamespace(language='es')
    with tempfile.TemporaryDirectory(prefix='acceptance-pcm-', dir='artifacts') as temporary:
        wav = Path(temporary) / 'non-speech-fixture.wav'
        with wave.open(str(wav), 'wb') as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
            stream.writeframes(b'\x01\x01' * 16000)
        _, provenance = recognize_audio(wav, 'es', backend=StubASR())
    report['audio'] = {'scope': 'ASR_STUB_GENERATED_NON_SPEECH_PCM_NOT_REAL_VOICE',
                       'dependencies': dependency_versions(), 'real_windows_microphone': 'PENDING_THIS_VERSION'}
    edited_text = 'El recorrido no fue malo.'

    for kind in kinds:
        seen = [observe(app, kind, row) for row in fixtures]
        long_text = ('Información del trayecto y sus paradas. ' * 150)[:5000]
        seen.append(observe(app, kind, {'id': 'BOUNDARY_5000', 'text': long_text}))
        report['models'][kind] = seen
        for row in seen:
            check(kind + ':' + row['id'], not row['violations'])
        for name, value in [('empty', ''), ('whitespace', ' \t\n'), ('null', None), ('too_long', 'x' * 5001)]:
            summary, encoded, _ = app.analyze_with_agents(value, kind)
            check(kind + ':reject_' + name, encoded == '{}' and 'Error' in summary)
        texts = [fixtures[0]['text'], fixtures[1]['text'], fixtures[0]['text'], fixtures[-1]['text']]
        records, table, counts, _, _ = app.run_batch_lines('\n'.join(texts), kind)
        check(kind + ':batch_count_order_duplicates', [r['texto_original'] for r in records] == texts and len(table) == 4 and sum(r[1] for r in counts) == 4)
        for record in records:
            require_effective(record, kind)
        original = deepcopy(records)
        edited, *_ = app.revise(records, records[0]['id'], 'neutral', '=1+1')
        check(kind + ':correction_nonmutating', records == original and edited[0]['sentimiento'] == original[0]['sentimiento'] and edited[0]['etiqueta_corregida_humano'] == 'neutral')
        json_path, csv_path, _ = app.export_files(edited)
        restored = json.loads(Path(json_path).read_text(encoding='utf-8'))['registros']
        exported = list(csv.DictReader(io.StringIO(Path(csv_path).read_bytes().decode('utf-8-sig'), newline='')))
        check(kind + ':json_roundtrip', restored == edited)
        check(kind + ':csv_count_model_and_formula', len(exported) == 4 and all(r['modelo_utilizado'] == kind for r in exported) and exported[0]['comentario_revision_humana'].startswith("'") and exported[-1]['texto_original'].startswith("'="))
        # Delete only files/directories created by this acceptance call.
        Path(json_path).unlink(); Path(csv_path).unlink(); Path(json_path).parent.rmdir()
        with tempfile.TemporaryDirectory(prefix='acceptance-csv-', dir='artifacts') as temporary:
            source = Path(temporary) / 'valid.csv'
            source.write_bytes('reseña,nota\r\n"Una frase, con coma\r\ny otra línea",fixture\r\n'.encode('utf-8-sig'))
            csv_records, *_ = app.run_batch_csv(str(source), kind)
            expected = 'Una frase, con coma\r\ny otra línea'
            check(kind + ':csv_callback_multiline', len(csv_records) == 1 and csv_records[0]['texto_original'] == expected)
            for record in csv_records:
                require_effective(record, kind)
            json_path, csv_path, _ = app.export_files(csv_records)
            roundtrip = list(csv.DictReader(io.StringIO(Path(csv_path).read_bytes().decode('utf-8-sig'), newline='')))
            check(kind + ':csv_export_original_crlf', len(roundtrip) == 1 and roundtrip[0]['texto_original'] == expected)
            Path(json_path).unlink(); Path(csv_path).unlink(); Path(json_path).parent.rmdir()
        _, encoded = app.analyze_audio_ui(edited_text, provenance, kind)
        audio_record = json.loads(encoded); require_effective(audio_record, kind)
        check(kind + ':editable_audio_stub', audio_record['procedencia_entrada']['fue_corregida'] is True and audio_record['procedencia_entrada']['transcripcion_original'] == 'El recorrido fue malo.' and audio_record['multiagente']['modalidad_percibida'] == 'audio_transcrito' and not record_violations(audio_record, edited_text))
        report['audio'][kind] = audio_record

    from .batch import parse_review_csv, parse_review_lines
    valid = b'\xef\xbb\xbfrese\xc3\xb1a,nota\n"Texto, con coma\ny otra l\xc3\xadnea",dato\n'
    check('csv_bom_quotes_multiline', parse_review_csv(valid) == ['Texto, con coma\ny otra línea'])
    invalid = {'duplicate_header': 'reseña,reseña\nprimera,segunda\n'.encode(),
               'extra_field': 'reseña\nprimera,perdida\n'.encode(),
               'missing_field': 'reseña,nota\nprimera\n'.encode(),
               'unclosed_quote': 'reseña\n"primera\n'.encode(),
               'bad_utf8': b'rese\xf1a\na\n', 'missing_header': b'other\na\n',
               'oversize': b'x' * (1024 * 1024 + 1)}
    for name, content in invalid.items():
        try:
            parse_review_csv(content)
            rejected = False
        except ValueError:
            rejected = True
        check('csv_reject_' + name, rejected)
    check('lines_boundary_200', len(parse_review_lines('\n'.join(['Fixture'] * 200))) == 200)
    try:
        parse_review_lines('\n'.join(['Fixture'] * 201))
        rejected = False
    except ValueError:
        rejected = True
    check('lines_reject_201', rejected)
    check('unknown_model_rejected', app.analyze('Texto funcional.', 'unknown')[1] == '{}')

    # Real invalid-artifact loading and actual LR inference; NOT a fake BETO.
    with tempfile.TemporaryDirectory(prefix='acceptance-fault-', dir='artifacts') as temporary:
        fault = Path(temporary)
        (fault / 'model_manifest.json').write_text('{}', encoding='utf-8')
        previous = os.environ.get('TRAVELER_BETO_ARTIFACT')
        os.environ['TRAVELER_BETO_ARTIFACT'] = str(fault.resolve())
        try:
            _, encoded = app.analyze(fixtures[0]['text'], 'beto')
            record = json.loads(encoded); require_effective(record, 'beto', fallback_expected=True)
            check('real_fallback_text_disclosed', not record_violations(record, fixtures[0]['text']))
            rows, *_, status = app.run_batch_lines('\n'.join(texts[:2]), 'beto')
            for row in rows:
                require_effective(row, 'beto', fallback_expected=True)
            check('real_fallback_batch_disclosed', len(rows) == 2 and 'respaldo' in status)
            report['fallback_observed'] = {'text': record, 'batch': rows}
            source = fault / 'valid.csv'
            source.write_text('reseña\nUna opinión funcional distinta.\n', encoding='utf-8')
            csv_records, *_ = app.run_batch_csv(str(source), 'beto')
            for row in csv_records:
                require_effective(row, 'beto', fallback_expected=True)
            check('real_fallback_csv_disclosed', len(csv_records) == 1)
            _, encoded = app.analyze_audio_ui(edited_text, provenance, 'beto')
            audio_record = json.loads(encoded); require_effective(audio_record, 'beto', fallback_expected=True)
            check('real_fallback_audio_disclosed', audio_record['procedencia_entrada']['fue_corregida'] is True)
            report['fallback_observed'].update(csv=csv_records, audio=audio_record)
        finally:
            if previous is None: os.environ.pop('TRAVELER_BETO_ARTIFACT', None)
            else: os.environ['TRAVELER_BETO_ARTIFACT'] = previous
    report['status'] = 'PASSED' if not report['violations'] else 'FAILED'
    return report


def run_predictive(commit, output, kinds=('lr', 'svm', 'beto'), *, reproduce=False):
    protocol, _, rows = registered_cases(commit)
    if tuple(kinds) != ('lr', 'svm', 'beto'):
        raise CorpusError('The registered primary comparison requires LR, SVM and actual BETO V3.')
    output = Path(output)
    if output.exists():
        raise CorpusError('Preserve the existing predictive report; use a new destination.')
    locks = Path('artifacts/academic-acceptance'); locks.mkdir(parents=True, exist_ok=True)
    lock = locks / ('observed-' + file_sha(REGISTERED[2]) + '.json')
    already = lock.exists() or PUBLISHED_PREDICTIVE.exists()
    if already and not reproduce:
        raise CorpusError('These cases are already observed; explicit reproduction required.')
    app = importlib.import_module('app')
    provenance = model_provenance(app, kinds)
    bindings = Path('docs/acceptance/2026-11-13-model-bindings.json')
    if not bindings.exists():
        raise CorpusError('Publish the fixed model bindings before the new measurement.')
    if not already and json.loads(bindings.read_text(encoding='utf-8'))['provenance'] != provenance:
        raise CorpusError('A model differs from its pre-measurement binding.')
    execution_commit = subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    report = {'schema': 'academic-predictive-v1', 'status': 'STARTED',
              'scope': 'OBSERVED_SOFTWARE_REPRODUCTION' if already else 'FIRST_REGISTERED_SYNTHETIC_MEASUREMENT',
              'registration_commit': commit, 'cases_sha256': file_sha(REGISTERED[2]),
              'execution_commit': execution_commit, 'pre_measurement_bindings_sha256': file_sha(bindings),
              'started_utc': datetime.now(timezone.utc).isoformat(), 'human_gold': False,
              'comparison_limit': protocol['comparison_limit'], 'authorship_limit': protocol['authorship_limit'],
              'model_or_threshold_selection': False, 'provenance': provenance, 'models': {},
              'source_sha256': source_fingerprints()}
    if not already:
        with lock.open('x', encoding='utf-8') as stream:
            json.dump({'registration_commit': commit, 'cases_sha256': report['cases_sha256'], 'provenance': provenance}, stream)
    start = time.perf_counter()
    write_report(output, report)
    try:
        for kind in kinds:
            observed = []
            report['models'][kind] = {'observations': observed}
            for row in rows:
                item = observe(app, kind, row)
                observed.append(item)
                if item['violations']:
                    raise CorpusError('Functional failure during prediction; do not score silently: ' + row['id'])
            labeled = [(r, o) for r, o in zip(rows, observed) if r['intended_label'] is not None]
            truth = [r['intended_label'] for r, _ in labeled]
            guesses = [o['record']['sentimiento'] for _, o in labeled]
            reviewed = [o['record']['revision_humana'] for _, o in labeled]
            metrics = summarize_predictions(truth, guesses, reviewed=reviewed)
            slices = {}
            for tag in sorted({t for r, _ in labeled for t in r['phenomena']}):
                indexes = [i for i, (r, _) in enumerate(labeled) if tag in r['phenomena']]
                slices[tag] = summarize_predictions([truth[i] for i in indexes], [guesses[i] for i in indexes], reviewed=[reviewed[i] for i in indexes])
            ambiguous = [o for r, o in zip(rows, observed) if r['intended_label'] is None]
            report['models'][kind].update(metrics=metrics, slices=slices,
                ambiguity={'n': len(ambiguous), 'referred': sum(o['record']['revision_humana'] for o in ambiguous), 'scope': 'AI_EXPECTED_REVIEW_NOT_HUMAN_GOLD'},
                errors=[{'id': r['id'], 'text': r['text'], 'intended_label': r['intended_label'], 'prediction': o['record']['sentimiento'], 'human_review': o['record']['revision_humana'], 'reasons': o['record']['multiagente']['razones_revision']} for r, o in labeled if r['intended_label'] != o['record']['sentimiento']],
                latency={'scope': 'APP_CALLBACK_WITH_PRELOADED_MODELS_NOT_COLD_START_OR_SLA', 'n': len(observed), 'p50_ms': float(np.percentile([o['latency_ms'] for o in observed], 50)), 'p95_ms': float(np.percentile([o['latency_ms'] for o in observed], 95))})
            write_report(output, report)
        report['status'] = 'COMPLETED'
    except Exception as exc:
        report.update(status='INCOMPLETE_OBSERVED_ATTEMPT', failure_type=type(exc).__name__)
        raise
    finally:
        report['wall_seconds'] = time.perf_counter() - start
        write_report(output, report)
    return report
