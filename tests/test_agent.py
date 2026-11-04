import pytest
from traveler_agent.agent import inspect_review
from traveler_agent.model import train
from traveler_agent.demo_data import DEMO
from traveler_agent.aspects import find_aspects,evidence_spans
from traveler_agent.policy import recommend
from traveler_agent.labels import normalize_label

@pytest.fixture(scope='module')
def model():
    return train([t for t,_ in DEMO],[y for _,y in DEMO])

def test_labels():
    assert normalize_label(' POSITIVE ')=='positivo'
    with pytest.raises(ValueError): normalize_label('2')

def test_prediction_structure(model):
    r=inspect_review('La habitación estaba sucia y el precio fue caro.',model)
    assert r['sentimiento'] in {'positivo','neutral','negativo'}
    assert 'higiene' in r['aspectos_exploratorios']
    assert 'precio' in r['aspectos_exploratorios']
    assert all(v >= 0 for v in r['puntuaciones_modelo_no_calibradas'].values())

def test_evidence_literal_only():
    t='La espera fue larga; el personal fue amable.'
    spans=evidence_spans(t,find_aspects(t))
    assert all(s in t for s in spans)

def test_review_gate(model):
    r=inspect_review('La visita fue muy buena.',model,threshold=1.01)
    assert r['revision_humana'] is True
    assert r['tipo_salida']=='revisión_humana'

def test_empty_review(model):
    with pytest.raises(ValueError): inspect_review('   ',model)

def test_no_negative_automation_on_review():
    assert recommend('negativo',{'seguridad':['robo']},True)[0]=='revisión_humana'
