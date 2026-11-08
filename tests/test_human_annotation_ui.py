"""UI adapters operate on synthetic test packets, never real contributions."""
import json
import re

import pytest

from traveler_agent.corpus import CorpusError
from traveler_agent.human_intake import (GUIDE,GUIDE_VERSION,load_packet,write_json,read_json,validate_consent)
from traveler_agent.participant_form import create_participant_form


def test_offline_form_is_private_unsigned_and_does_not_register_people(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    output=tmp_path/'data/private/delivery/form.html'
    result=create_participant_form(output)
    page=output.read_text(encoding='utf-8')
    payload=json.loads(re.search(r'<script id="consent-template" type="application/json">(.*?)</script>',page,re.S).group(1))
    assert result['status']=='OFFLINE_FORM_READY_NOT_CONSENT_OR_REVIEW'
    assert payload['author_id']==result['author_id']
    assert payload['public_raw_text'] is False and payload['annotation'] is False
    with pytest.raises(CorpusError):validate_consent(payload)
    assert "default-src 'none'" in page and "form-action 'none'" in page
    assert not re.search(r'<script[^>]+src\s*=|fetch\s*\(|XMLHttpRequest|localStorage',page)
    assert not (tmp_path/'data/private/piloto01-human-v1/sources.csv').exists()
    with pytest.raises(FileExistsError):create_participant_form(output)
    with pytest.raises(CorpusError):create_participant_form(tmp_path/'public.html')


def test_packet_ui_never_reads_master_corpus_or_other_annotations(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    import corpus_annotation_app as ui
    def forbidden(*_):pytest.fail('Blind packet mode must never read the master corpus')
    monkeypatch.setattr(ui,'available_reviews',forbidden)
    monkeypatch.setattr(ui,'read_review',forbidden)
    monkeypatch.setattr(ui,'quality_preview',forbidden)
    packet=tmp_path/'data/private/ann/task.json'
    answers=tmp_path/'data/private/ann/answers.json'
    write_json(packet,{'schema':'blind-human-task-v1','task_id':'task_fixture','annotator_id':'ann_fixture',
                      'guide_version':GUIDE_VERSION,'guide':GUIDE,'items':[{'item_id':'item_fixture','text':'Texto artificial para UI privada.'}]})
    original,load,save,seal=ui.packet_callbacks(packet,answers)
    assert load('item_fixture')[:2]==('Texto artificial para UI privada.',None)
    assert 'guardado' in save('item_fixture','neutral',False,False,'incierto','ausente','{}','[]')
    assert load('item_fixture')[1]=='neutral'
    assert load('item_fixture')[4:6]==('incierto','ausente')
    assert 'finalizadas' in seal(True)
    assert 'No guardado' in save('item_fixture','positivo',False,False,False,False,'{}','[]')
    app=ui.build_packet_app(packet,answers)
    assert app is not None and read_json(answers)['independence_attested'] is True


def test_foreign_or_label_enriched_packets_are_rejected(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    path=tmp_path/'data/private/task.json'
    packet={'schema':'blind-human-task-v1','task_id':'task_fixture','annotator_id':'ann',
            'guide_version':GUIDE_VERSION,'guide':GUIDE,'items':[{'item_id':'i','text':'Fixture inventado.','sentiment':'positivo'}]}
    write_json(path,packet)
    with pytest.raises(CorpusError,match='solo admite'):load_packet(path)
