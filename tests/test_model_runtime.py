"""Optional adapter isolation, real fallback execution and exported provenance."""
import csv
import io
import json
from types import SimpleNamespace
import wave

import numpy as np
import pytest

import app
from traveler_agent import model_runtime as runtime
from traveler_agent.batch import export_csv


class FixtureClassifier:
    classes_=np.array(['negativo','neutral','positivo'])
    metadata={'revision':'fixture-not-real-BETO','training':{'mode':'fixture_supervised_contract'}}
    def predict_proba(self,texts):return np.tile([.1,.7,.2],(len(texts),1))


@pytest.fixture
def configured(tmp_path,monkeypatch):
    path=tmp_path/'model';path.mkdir();(path/'model_manifest.json').write_text('{}')
    monkeypatch.setattr(runtime,'configured_artifact',lambda:path)
    return path


def test_classical_default_never_imports_or_loads_optional_backend(monkeypatch):
    monkeypatch.setattr(runtime,'_load',lambda *_:pytest.fail('classical path attempted BETO'))
    record,effective=runtime.run_selected('lr',app.get_demo_model,lambda m:{'scores':m.predict_proba(['Fixture.']).tolist()})
    assert effective=='lr' and 'seleccion_modelo' not in record
    assert 'beto' not in app.MODEL_KINDS


def test_missing_optional_artifact_runs_actual_lr_and_discloses_fallback(monkeypatch):
    monkeypatch.setattr(runtime,'configured_artifact',lambda:None)
    _,encoded=app.analyze('El guía fue amable y disfruté la visita.','beto')
    record=json.loads(encoded)
    assert record['modelo_utilizado']=='lr'
    assert record['seleccion_modelo']['requested']=='beto'
    assert record['seleccion_modelo']['fallback'] is True
    assert record['multiagente']['orquestador']=='TravelerOrchestrator'


def test_successful_optional_backend_is_executed_for_text_batch_and_exports(configured,monkeypatch):
    calls=[]
    model=FixtureClassifier()
    original=model.predict_proba
    def counted(texts):calls.extend(texts);return original(texts)
    model.predict_proba=counted
    monkeypatch.setattr(runtime,'_load',lambda *_:model)
    summary,encoded=app.analyze('Texto artificial de prueba.','beto')
    record=json.loads(encoded)
    assert record['sentimiento']=='neutral' and record['modelo_utilizado']=='beto'
    assert not record['seleccion_modelo']['fallback'] and len(calls)==1
    assert 'experimental' in summary
    rows,*_=app.run_batch_lines('Primer fixture artificial.\nOtro fixture distinto.','beto')
    assert len(rows)==2 and len(calls)==3
    parsed=list(csv.DictReader(io.StringIO(export_csv(rows))))
    assert parsed[0]['modelo_utilizado']==parsed[0]['modelo_solicitado']=='beto'
    assert parsed[0]['fallback_modelo']=='False' and parsed[0]['version_modelo']


def test_runtime_inference_failure_retries_whole_batch_and_names_effective_model(configured,monkeypatch):
    class Broken(FixtureClassifier):
        def predict_proba(self,texts):raise RuntimeError('private diagnostic must not be exported')
    monkeypatch.setattr(runtime,'_load',lambda *_:Broken())
    rows,*_,status=app.run_batch_lines('Servicio amable.\nHabitación sucia.','beto')
    assert len(rows)==2 and 'respaldo' in status
    assert {r['modelo_utilizado'] for r in rows}=={'lr'}
    assert all(r['seleccion_modelo']['fallback'] for r in rows)
    assert 'private diagnostic' not in json.dumps(rows)
    exported=list(csv.DictReader(io.StringIO(export_csv(rows))))
    assert all(r['modelo_solicitado']=='beto' and r['fallback_modelo']=='True' for r in exported)


def test_unknown_model_cannot_be_silently_relabelled_as_classical():
    _,encoded=app.analyze('Fixture artificial.','not-a-model')
    assert encoded=='{}'


def test_optional_classifier_keeps_editable_transcript_provenance_with_asr_stub(configured,monkeypatch,tmp_path):
    # Generated non-speech PCM + explicit ASR stub: not a real voice test.
    from traveler_agent.audio_asr import recognize_audio
    class ASR:
        def transcribe(self,*args,**kwargs):
            return iter([SimpleNamespace(text='Fue mala la visita.')]),SimpleNamespace(language='es')
    audio=tmp_path/'fixture.wav'
    with wave.open(str(audio),'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000)
        stream.writeframes(b'\x01\x01'*16000)
    _,provenance=recognize_audio(audio,'es',backend=ASR())
    monkeypatch.setattr(runtime,'_load',lambda *_:FixtureClassifier())
    _,encoded=app.analyze_audio_ui('No fue mala la visita.',provenance,'beto')
    record=json.loads(encoded)
    assert record['modelo_utilizado']=='beto' and not record['seleccion_modelo']['fallback']
    assert record['procedencia_entrada']['transcripcion_original']=='Fue mala la visita.'
    assert record['procedencia_entrada']['fue_corregida'] is True
