"""Acceptance contracts, never gold-label fitting or a pretend real BETO."""
import csv
import io
import json
from copy import deepcopy

import pytest

import app
from traveler_agent.academic_acceptance import registered_cases, require_effective, record_violations
from traveler_agent.batch import parse_review_csv
from traveler_agent.corpus import CorpusError

REGISTRATION = 'e20326484aa33d335e85bf0f8551c24adbbd13e1'


@pytest.mark.parametrize('content', [
    'reseña,reseña\nprimera,segunda\n',
    'reseña,nota,nota\nprimera,a,b\n',
    'reseña\nprimera,perdida\n',
    'reseña,nota\nprimera\n',
    'reseña\n"sin cierre\n',
    'reseña,nota\n"cerrada"basura,a\n',
])
def test_ambiguous_csv_structure_rejected_before_any_model(content, tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        parse_review_csv(content.encode('utf-8'))
    path = tmp_path / 'bad.csv'; path.write_text(content, encoding='utf-8')
    monkeypatch.setattr(app, 'get_demo_model', lambda *args: pytest.fail('Invalid CSV reached inference'))
    records, table, counts, _, message = app.run_batch_csv(str(path), 'beto')
    assert records == [] and table == [] and 'Error' in message


def test_csv_keeps_bom_commas_multiline_extra_columns_order_and_duplicates():
    text = 'reseña,nota\r\n"Frase, con coma\ny salto",a\r\nigual,b\r\nigual,c\r\n'
    assert parse_review_csv(text.encode('utf-8-sig')) == ['Frase, con coma\ny salto', 'igual', 'igual']


def test_csv_preserves_original_crlf_inside_quoted_review():
    assert parse_review_csv('reseña\r\n"Primera línea\r\nsegunda línea"\r\n'.encode('utf-8-sig')) == ['Primera línea\r\nsegunda línea']


def test_export_preserves_multiline_csv_under_windows_newline_defaults(tmp_path, monkeypatch):
    from pathlib import Path
    from traveler_agent.batch import analyze_batch
    original_write = Path.write_text
    def windows_defaults(path, data, encoding=None, errors=None, newline=None):
        if path.suffix == '.csv' and newline is None:
            newline = '\r\n'
        return original_write(path, data, encoding=encoding, errors=errors, newline=newline)
    monkeypatch.setattr(Path, 'write_text', windows_defaults)
    text = 'Primera línea\r\nsegunda línea'
    records = analyze_batch([text], app.get_demo_model('lr'))
    json_path, csv_path, _ = app.export_files(records)
    try:
        content = Path(csv_path).read_bytes().decode('utf-8-sig')
        exported = list(csv.DictReader(io.StringIO(content, newline='')))
        assert exported[0]['texto_original'] == text
        assert json.loads(Path(json_path).read_text(encoding='utf-8'))['registros'][0]['texto_original'] == text
    finally:
        Path(json_path).unlink(); Path(csv_path).unlink(); Path(json_path).parent.rmdir()


def test_effective_model_guard_never_counts_lr_fallback_as_beto():
    fallback = {'modelo_utilizado': 'lr', 'seleccion_modelo': {'requested': 'beto', 'effective': 'lr', 'fallback': True}}
    with pytest.raises(CorpusError, match='did not execute'):
        require_effective(fallback, 'beto')
    require_effective(fallback, 'beto', fallback_expected=True)
    with pytest.raises(CorpusError):
        require_effective({'modelo_utilizado': 'beto'}, 'beto')


def test_registered_groups_and_labels_are_frozen_in_git():
    protocol, functional, predictive = registered_cases(REGISTRATION)
    assert protocol['threshold_changes_forbidden'] is True
    assert len(functional) == 13 and len(predictive) == 44
    assert sum(row['intended_label'] is None for row in predictive) == 8
    with pytest.raises(CorpusError): registered_cases('not-a-commit')


def test_auditor_detects_missing_agent_and_unjustified_review():
    text = 'Fixture funcional para verificar los contratos de coordinación.'
    _, encoded = app.analyze(text, 'lr'); record = json.loads(encoded)
    assert not record_violations(record, text)
    broken = deepcopy(record); broken['multiagente']['traza'].pop()
    assert 'trace_sequence_invalid' in record_violations(broken, text)
    broken = deepcopy(record); broken['multiagente']['razones_revision'] = ['INVENTED_REASON']
    assert 'review_rule_or_reasons_inconsistent' in record_violations(broken, text)
