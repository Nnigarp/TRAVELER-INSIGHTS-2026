"""Synthetic plumbing fixtures only: no real consent, authors or human scores."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from traveler_agent.corpus import CorpusError, _read_csv, _write_csv, SOURCE_COLUMNS, REVIEW_COLUMNS, ANNOTATION_COLUMNS, file_sha
from traveler_agent import human_intake as workflow
from traveler_agent.human_intake_cli import run


@pytest.fixture
def service(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    result=workflow.HumanIntake(Path('data/private/fixture-human'))
    result.init()
    return result


def fixture_author(service,code='author_fixture'):
    source=service.root/'incoming'/f'{code}.json'
    value=workflow.new_consent_form()
    value.update(author_id=code,accepted_at='2026-11-07T12:00:00+00:00')
    for key in ('annotation','training','evaluation','aggregate_results','original_authorship','real_experience'): value[key]=True
    workflow.write_json(source,value)
    return source,service.register_consent(source,'collector_fixture')['author_id']


def fixture_review(service,author,text='Texto artificial para comprobar el gestor, sin reseña humana real.',place='place_fixture',related=()):
    path=service.root/'incoming'/f'text_{hashlib.sha256(text.encode()).hexdigest()}.txt'
    if not path.exists(): path.write_text(text,encoding='utf-8')
    return service.contribute(path,author,place,'collector_fixture',related,
        privacy_reviewed=True,spanish_verified=True,original_human=True,real_experience=True)['review_id']


def judgment(label='neutral',**changes):
    return {'sentiment':label,'aspects':{},'spans':[],'mixed':False,'ambiguous':label=='desconocido','negation':False,'contrast':False,**changes}


def fixture_task(service,person,labels,*,imported=True):
    packet_path=service.root/'delivery'/f'{person}.json'
    returned=service.root/'incoming'/f'{person}-answers.json'
    task=service.create_task(person,packet_path,100)
    index=workflow.read_json(service.root/'tasks'/f"{task['task_id']}.index.json")
    for item in workflow.load_packet(packet_path)['items']:
        rid=index['mapping'][item['item_id']]['review_id']
        value=labels[rid]
        workflow.save_answer(packet_path,returned,item['item_id'],value if isinstance(value,dict) else judgment(value))
    workflow.seal_answers(packet_path,returned,independent=True)
    if imported: service.import_answers(returned)
    return packet_path,returned,task


def test_empty_initialization_is_idempotent_and_is_not_admission(service):
    assert service.status()['active_reviews']==0
    assert service.status()['registered_consents']==0
    hashes={p.name:file_sha(p) for p in service.root.glob('*.csv')}
    service.init()
    assert hashes=={p.name:file_sha(p) for p in service.root.glob('*.csv')}
    with pytest.raises(CorpusError,match='Sin reseñas humanas'):
        service.freeze(service.root/'snapshot','collector',links_reviewed=True)


def test_blank_consent_is_not_approval_and_real_schema_is_required(service):
    source=service.root/'incoming/blank.json'
    workflow.consent_template(source)
    with pytest.raises(CorpusError,match='aceptación'):
        service.register_consent(source,'collector')
    assert _read_csv(service.root/'sources.csv',SOURCE_COLUMNS)==[]
    valid=workflow.read_json(source)
    valid.update(accepted_at='2026-11-07T12:00:00Z')
    for key in ('annotation','training','evaluation','aggregate_results','original_authorship','real_experience'): valid[key]=True
    for changed in ({'notice_sha256':'0'*64},{'email':'not-collected@example.invalid'},{'public_raw_text':True},{'accepted_at':'2099-01-01T00:00:00Z'}):
        with pytest.raises(CorpusError): workflow.validate_consent({**valid,**changed})


def test_consent_retry_is_idempotent_and_proof_tampering_blocks_use(service):
    source,author=fixture_author(service)
    first=service._receipt(author)
    assert service.register_consent(source,'collector_fixture')['consent_id']==first['consent_id']
    proof=service.root/'permissions'/f"{first['consent_id']}.proof.json"
    proof.write_text('{}')
    with pytest.raises(CorpusError,match='modificada'):
        fixture_review(service,author)
    assert _read_csv(service.root/'reviews.csv',REVIEW_COLUMNS)==[]


def test_private_paths_resolve_symlink_escape_before_writing(service,tmp_path):
    outside=tmp_path/'public';outside.mkdir()
    alias=service.root/'escape';alias.symlink_to(outside,target_is_directory=True)
    with pytest.raises(CorpusError): workflow.consent_template(alias/'proof.json')
    with pytest.raises(CorpusError): workflow.HumanIntake(outside/'private')
    assert list(outside.iterdir())==[]


def test_intake_checks_and_receipt_are_required(service):
    _,author=fixture_author(service)
    text=service.root/'incoming/text.txt';text.write_text('Texto de fixture artificial.',encoding='utf-8')
    with pytest.raises(CorpusError,match='privacidad'):
        service.contribute(text,author,'place','collector')
    with pytest.raises(CorpusError):
        service.contribute(text,'author_unregistered','place','collector',privacy_reviewed=True,spanish_verified=True,original_human=True,real_experience=True)
    assert _read_csv(service.root/'reviews.csv',REVIEW_COLUMNS)==[]


def test_two_blind_packets_and_third_person_adjudication_with_agreement(service):
    _,author=fixture_author(service)
    one=fixture_review(service,author,'Fixture artificial alfa con dos valoraciones opuestas.')
    two=fixture_review(service,author,'Registro de tubería beta, descripción sin valoración.',place='another_place')
    first,answers,task=fixture_task(service,'annA',{one:'positivo',two:'neutral'})
    with pytest.raises(CorpusError,match='dos dictámenes'):
        service.adjudication_view(one)
    with pytest.raises(CorpusError,match='pendientes'):
        service.create_task('annA',service.root/'delivery/duplicate.json')
    second,_,_=fixture_task(service,'annB',{one:'negativo',two:'neutral'})
    packet=workflow.read_json(second)
    assert set(packet)=={'schema','task_id','annotator_id','guide_version','guide','items'}
    assert all(set(item)=={'item_id','text'} for item in packet['items'])
    serialized=json.dumps(packet)
    assert author not in serialized and one not in serialized and 'place_fixture' not in serialized and 'annA' not in serialized
    before=service.status()
    assert before['reviews_with_two_verified_annotations']==2
    assert before['raw_agreement'][0]['raw_agreement']==.5
    assert before['agreement']['comparisons'][0]['shared_reviews']==2
    with pytest.raises(CorpusError,match='tercera'):
        service.adjudicate(one,'annA',judgment('desconocido',mixed=True),'Fixture disagreement')
    service.adjudicate(one,'judge',judgment('desconocido',mixed=True),'Fixture: no predominio global')
    service.adjudicate(two,'judge',judgment('neutral'),'Fixture descriptivo')
    after=service.status()
    assert after['adjudicated_label_counts']=={'desconocido':1,'neutral':1}
    assert after['mixed_reviews']==after['ambiguous_reviews']==1
    assert 'Fixture artificial' not in json.dumps(after)
    assert workflow.read_json(answers)['independence_attested'] is True
    with pytest.raises(CorpusError,match='finalizadas'):
        workflow.save_answer(first,answers,workflow.read_json(first)['items'][0]['item_id'],judgment())


@pytest.mark.parametrize('value',[judgment('neutral',mixed=True),judgment('neutral',ambiguous=True),judgment('desconocido',ambiguous=False),judgment('positivo',negation='yes')])
def test_neutral_mixed_ambiguous_and_phenomena_are_not_conflated(value):
    with pytest.raises(CorpusError): workflow.validate_judgment(value,'Texto artificial de prueba.')


def test_incomplete_or_unsealed_answers_and_changed_packets_are_refused(service):
    _,author=fixture_author(service);rid=fixture_review(service,author)
    packet=service.root/'delivery/a.json';service.create_task('ann',packet)
    answers=service.root/'incoming/a-answers.json'
    with pytest.raises(CorpusError,match='Completar'):
        workflow.seal_answers(packet,answers,independent=True)
    item=workflow.read_json(packet)['items'][0]
    workflow.save_answer(packet,answers,item['item_id'],judgment())
    with pytest.raises(CorpusError,match='finalizadas'):
        service.import_answers(answers)
    workflow.seal_answers(packet,answers,independent=True)
    value=workflow.read_json(answers);value['packet_sha256']='0'*64
    workflow.write_json(answers,value,exclusive=False)
    with pytest.raises(CorpusError): service.import_answers(answers)
    assert _read_csv(service.root/'annotations.csv',ANNOTATION_COLUMNS)==[]


def test_import_resumes_without_overwriting_after_interruption(service,monkeypatch):
    _,author=fixture_author(service)
    labels={fixture_review(service,author,f'Fixture artificial con huella {hashlib.sha256(str(i).encode()).hexdigest()}'):'neutral' for i in range(3)}
    _,answers,_=fixture_task(service,'ann',labels,imported=False)
    original=workflow.record_annotation
    calls=[]
    def interrupted(*args,**kwargs):
        calls.append(True)
        if len(calls)==2: raise OSError('simulated')
        return original(*args,**kwargs)
    monkeypatch.setattr(workflow,'record_annotation',interrupted)
    with pytest.raises(OSError): service.import_answers(answers)
    with pytest.raises(CorpusError,match='incompleta'): service.status()
    monkeypatch.setattr(workflow,'record_annotation',original)
    service.import_answers(answers);service.import_answers(answers)
    assert len(_read_csv(service.root/'annotations.csv',ANNOTATION_COLUMNS))==3
    assert service.status()['annotation_rows']==3


def test_withdrawal_blocks_pending_import_and_future_tasks(service):
    _,author=fixture_author(service);rid=fixture_review(service,author)
    _,answers,task=fixture_task(service,'ann',{rid:'neutral'},imported=False)
    service.withdraw(author)
    with pytest.raises(CorpusError,match='retirado'): service.import_answers(answers)
    assert service.status()['active_reviews']==0
    with pytest.raises(CorpusError,match='pendientes'): service.create_task('other',service.root/'delivery/other.json')
    service.cancel_task(task['task_id'])


def test_connected_components_use_author_or_place_or_related_or_exact_without_deletion():
    texts=['alpha unique 91827','beta unique 76123','gamma unique 67291','delta unique 19276','delta unique 19276','another long phrase with the same words until the ending now','another long phrase with the same words until the ending later']
    reviews=[{'review_id':str(i),'text':text} for i,text in enumerate(texts)]
    links=[{'review_id':str(i),'author_id':f'a{i}','place_id':f'p{i}','related_ids_json':'[]'} for i in range(7)]
    links[1]['author_id']=links[0]['author_id'];links[2]['place_id']=links[1]['place_id']
    links[3]['related_ids_json']='["2"]'
    groups,candidates=workflow.connected_groups(reviews,links)
    assert len({groups[str(i)] for i in range(5)})==1
    assert groups['5']==groups['6'] and groups['0']!=groups['5']
    assert candidates['candidates'] and len(groups)==len(reviews)


def test_cli_status_never_emits_raw_text_or_permissions(service,capsys):
    base=['--root',str(service.root)]
    source,author=fixture_author(service);fixture_review(service,author)
    status=run(base+['status'])
    assert status['active_reviews']==1 and status['adjudicated_reviews']==0
    assert 'Texto artificial' not in json.dumps(status) and 'proof_sha256' not in json.dumps(status)
    assert workflow.NOTICE_VERSION in run(base+['notice'])['version']


def test_private_snapshot_reuses_existing_splits_and_blocks_stale_consent(service,monkeypatch):
    # Labels and approvals here are deliberately simulated software fixtures.
    # Three fixture labels per place guarantee support without tuning a seed.
    labels={};authors=[]
    for group in range(12):
        _,author=fixture_author(service,f'author_fixture_{group}')
        authors.append(author)
        for index,label in enumerate(('negativo','neutral','positivo')):
            unique=hashlib.sha256(f'fixture-{group}-{index}'.encode()).hexdigest()
            rid=fixture_review(service,author,f'Caso ficticio de prueba {unique}',f'place_fixture_{group}')
            labels[rid]=label
    extra=fixture_review(service,authors[0],'Caso ficticio sin predominio para comprobar exclusión de ambiguos.','another_fixture_place')
    labels[extra]=judgment('desconocido',mixed=True)
    fixture_task(service,'annA',labels);fixture_task(service,'annB',labels)
    for rid,label in labels.items():
        service.adjudicate(rid,'judge',label if isinstance(label,dict) else judgment(label),'Fixture explícito, no anotación humana real')
    import traveler_agent.scientific_benchmark as scientific
    monkeypatch.setattr(scientific,'_new_estimator',lambda *_:pytest.fail('No model fit is allowed in intake/freeze'))
    output=service.root/'snapshots/v1'
    result=service.freeze(output,'collector',links_reviewed=True)
    assert result['ambiguous_excluded']==1
    assert result['components']==12
    assert sum(part['rows'] for part in result['split_summary'].values())==36
    assert service.verify_snapshot(output)['status'].startswith('SNAPSHOT_AND_CURRENT')
    manifest=workflow.read_json(output/'human-snapshot.json')
    assigned={}
    from traveler_agent.corpus_splits import TRAIN_COLUMNS
    for split in ('train','validation','test'):
        for row in _read_csv(output/'splits'/f'{split}.csv',TRAIN_COLUMNS):
            assigned[row['review_id']]=split
    canonical=_read_csv(service.root/'reviews.csv',REVIEW_COLUMNS)
    for author in authors:
        assert len({assigned[r['review_id']] for r in canonical if r['source_id']==author and r['review_id']!=extra})==1
    assert extra not in assigned
    assert len(manifest['component_map'])==37
    with pytest.raises(CorpusError,match='existente'): service.freeze(output,'collector',links_reviewed=True)
    received=next((service.root/'received').glob('*.json'))
    original=workflow.read_json(received)
    changed=deepcopy(original)
    first_item=next(iter(changed['answers']))
    changed['answers'][first_item]['negation']=True
    workflow.write_json(received,changed,exclusive=False)
    with pytest.raises(CorpusError,match='Pruebas de anotación'):service.verify_snapshot(output)
    workflow.write_json(received,original,exclusive=False)
    service.verify_snapshot(output)
    original_splitter=workflow.prepare_splits
    original_links=(service.root/'intake_links.csv').read_bytes()
    def changed_during_freeze(*args,**kwargs):
        result=original_splitter(*args,**kwargs)
        with (service.root/'intake_links.csv').open('ab') as stream:stream.write(b'\n')
        return result
    monkeypatch.setattr(workflow,'prepare_splits',changed_during_freeze)
    with pytest.raises(CorpusError,match='durante el congelado'):
        service.freeze(service.root/'snapshots/racy','collector',links_reviewed=True)
    assert not (service.root/'snapshots/racy').exists()
    (service.root/'intake_links.csv').write_bytes(original_links)
    monkeypatch.setattr(workflow,'prepare_splits',original_splitter)
    service.withdraw(authors[0])
    with pytest.raises(CorpusError,match='retirado'): service.verify_snapshot(output)


def test_interrupted_intake_cannot_silently_enter_annotation_or_freeze(service,monkeypatch):
    _,author=fixture_author(service)
    original=workflow._write_csv
    def interrupted(path,*args):
        if path.name=='intake_links.csv': raise OSError('simulated interrupted sidecar')
        return original(path,*args)
    monkeypatch.setattr(workflow,'_write_csv',interrupted)
    with pytest.raises(OSError): fixture_review(service,author)
    with pytest.raises(CorpusError,match='trazabilidad'): service.status()


def test_unicode_spans_survive_blind_import_and_match_existing_annotation_tools(service):
    _,author=fixture_author(service)
    text='Fixture: la atención fue paciente.'
    rid=fixture_review(service,author,text)
    start=text.index('paciente')
    value=judgment('positivo',aspects={'servicio':'positivo'},spans=[{'aspect':'servicio','polarity':'positivo','start':start,'end':start+8}])
    fixture_task(service,'ann', {rid:value})
    rows=_read_csv(service.root/'evidence_spans.csv',workflow.SPAN_COLUMNS)
    assert rows[0]['text']=='paciente' and int(rows[0]['start'])==start


def test_nested_permission_directory_cannot_escape_private_root(service,tmp_path):
    outside=tmp_path/'outside';outside.mkdir()
    (service.root/'permissions').symlink_to(outside,target_is_directory=True)
    with pytest.raises(CorpusError,match='resuelta'):fixture_author(service)
    assert list(outside.iterdir())==[]


def test_legacy_annotation_without_blind_packet_proof_is_not_admitted(service):
    _,author=fixture_author(service);rid=fixture_review(service,author)
    workflow.record_annotation(service.root/'annotations.csv',{rid},review_id=rid,annotator_id='legacy',sentiment='neutral',aspects={})
    with pytest.raises(CorpusError,match='sin prueba'):service.status()


def test_uncertain_phenomena_are_distinct_from_global_ambiguity(service):
    _,author=fixture_author(service);rid=fixture_review(service,author)
    value=judgment('positivo',negation=None,contrast=None)
    fixture_task(service,'annA',{rid:value});fixture_task(service,'annB',{rid:value})
    service.adjudicate(rid,'judge',value,'Fixture: global claro, fenómeno incierto')
    report=service.status()
    assert report['ambiguous_reviews']==0
    assert report['human_phenomena_uncertain']=={'negation':1,'contrast':1}
    assert report['human_phenomena_counts']=={'negation':0,'contrast':0}


def test_local_draft_and_seal_share_a_single_writer_lock(service):
    _,author=fixture_author(service);fixture_review(service,author)
    packet=service.root/'delivery/locking.json';service.create_task('ann',packet)
    output=service.root/'incoming/locking-answers.json'
    iid=workflow.load_packet(packet)['items'][0]['item_id']
    with workflow.coordinator_lock(output.parent):
        with pytest.raises(CorpusError,match='ocupado'):workflow.save_answer(packet,output,iid,judgment())
        with pytest.raises(CorpusError,match='ocupado'):workflow.seal_answers(packet,output,independent=True)
    assert not output.exists()
    workflow.save_answer(packet,output,iid,judgment())
    workflow.seal_answers(packet,output,independent=True)


@pytest.mark.parametrize('mutation',['person','text_binding','duplicate_review'])
def test_private_assignment_cannot_reattribute_or_misbind_human_judgments(service,mutation):
    _,author=fixture_author(service)
    one=fixture_review(service,author,'Fixture alfa de mapeo privado de dictámenes.')
    two=fixture_review(service,author,'Fixture beta: otro contenido para otra reseña asignada.')
    packet,answers,task=fixture_task(service,'ann',{one:'neutral',two:'neutral'},imported=False)
    path=service.root/'tasks'/f"{task['task_id']}.index.json"
    index=workflow.read_json(path)
    items=list(index['mapping'])
    if mutation=='person':index['annotator_id']='other_person'
    elif mutation=='text_binding':index['mapping'][items[0]]['text_sha256']='0'*64
    else:index['mapping'][items[1]]['review_id']=index['mapping'][items[0]]['review_id']
    workflow.write_json(path,index,exclusive=False)
    with pytest.raises(CorpusError):service.import_answers(answers)
    assert _read_csv(service.root/'annotations.csv',ANNOTATION_COLUMNS)==[]
