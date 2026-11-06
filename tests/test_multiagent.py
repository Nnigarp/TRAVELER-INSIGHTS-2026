"""Real agent-to-agent delegation / state / failure integration tests."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from traveler_agent.agent import inspect_review
from traveler_agent.multiagent import (
    AgentFailure, TravelerOrchestrator, PerceptionAgent, SemanticsAgent,
    ControlAgent, ActionAgent, Message,
)
from traveler_agent.demo_data import DEMO
from traveler_agent.model import train
from traveler_agent.batch import analyze_batch


@pytest.fixture(scope="module")
def model():
    return train([x for x, _ in DEMO], [label for _, label in DEMO])


class Spy:
    def __init__(self, inner):
        self.inner = inner
        self.messages = []

    def handle(self, message):
        self.messages.append(message)
        return self.inner.handle(message)


def _spies(model, *, threshold=.56):
    agents = {
        "percepcion": Spy(PerceptionAgent()),
        "semantica": Spy(SemanticsAgent(model)),
        "control": Spy(ControlAgent(threshold)),
        "accion": Spy(ActionAgent()),
    }
    return agents


def test_orchestrator_delegates_four_distinct_tasks_and_validates_handoffs(model):
    spies = _spies(model)
    result = TravelerOrchestrator(model, agents=spies).run(
        "El guía fue amable, pero esperamos demasiado para entrar.",
        review_id="tour-01",
    )
    assert result["id"] == "tour-01"
    assert result["sentimiento"] in {"positivo", "neutral", "negativo"}
    assert set(result["puntuaciones_modelo_no_calibradas"]) == {
        "positivo", "neutral", "negativo"
    }
    assert "servicio" in result["aspectos_exploratorios"]
    assert "espera" in result["aspectos_exploratorios"]
    assert result["revision_humana"] is True
    assert result["tipo_salida"] == "revisión_humana"
    assert result["multiagente"]["estado"] == "REVISION_HUMANA"
    assert result["multiagente"]["plan"] == [
        "percepcion", "semantica", "control", "accion"
    ]
    assert [name for name in spies if len(spies[name].messages) == 1] == list(spies)
    assert spies["percepcion"].messages[0].sender == "orquestador"
    assert spies["semantica"].messages[0].payload["text"].startswith("El guía")
    assert "semantic" in spies["control"].messages[0].payload
    assert set(spies["accion"].messages[0].payload) == {"semantic", "control"}
    trace = result["multiagente"]["traza"]
    assert len(trace) == 10
    assert [t["sequence"] for t in trace] == list(range(1, 11))
    assert [t["agent"] for t in trace if t["status"] == "OK"] == list(spies)
    assert all("El guía" not in json.dumps(t, ensure_ascii=False) for t in trace)
    assert result["multiagente"]["acciones_externas"] is False


def test_non_review_route_when_policy_allows(model):
    r = TravelerOrchestrator(model, threshold=0).run(
        "La visita fue excelente y todo salió bien."
    )
    assert r["revision_humana"] is False
    assert r["multiagente"]["estado"] == "COMPLETADO"
    assert r["tipo_salida"] in {"informativo", "sugerencia"}


def test_legacy_entrypoint_and_batch_keep_original_contract(model):
    r = inspect_review("La habitación estaba sucia y el precio fue caro.", model)
    for key in (
        "id", "sentimiento", "puntuaciones_modelo_no_calibradas",
        "confianza_no_calibrada", "aspectos_exploratorios",
        "perfil_linguistico_exploratorio",
        "fragmentos_literales_no_explicativos", "revision_humana",
        "tipo_salida", "recomendacion", "advertencia",
    ):
        assert key in r
    batch = analyze_batch(["El servicio fue pésimo.", "Una visita bastante agradable."], model)
    assert len(batch) == 2
    assert all(row["multiagente"]["traza"] for row in batch)
    assert [r["id"] for r in batch] == ["demo-001", "demo-002"]


def test_audio_modality_trace_is_metadata_only(model):
    r = inspect_review(
        "Un hotel con habitaciones limpias.", model,
        modality="audio_transcrito", edited_by_human=True,
    )
    assert r["multiagente"]["modalidad_percibida"] == "audio_transcrito"
    assert r["multiagente"]["transcripcion_editada"] is True
    assert "habitaciones limpias" not in json.dumps(r["multiagente"]["traza"])
    assert r["multiagente"]["memoria"].startswith("episodica_solo_sesion")


def test_fails_closed_with_invalid_input_or_classifier_output(model):
    with pytest.raises(ValueError, match="detenido"):
        inspect_review("", model)

    class Broken:
        classes_ = np.array(["positivo", "neutral", "negativo"])

        def predict_proba(self, _):
            return np.array([[float("nan"), .4, .6]])

    with pytest.raises(AgentFailure, match="detenido"):
        TravelerOrchestrator(Broken()).run("Entrada válida")

    class Throws:
        classes_ = np.array(["positivo", "neutral", "negativo"])

        def predict_proba(self, _):
            raise RuntimeError("Sensitive private text NEVER to log")

    with pytest.raises(AgentFailure, match="detenido") as exc:
        TravelerOrchestrator(Throws()).run("Entrada privada")
    assert "Sensitive" not in str(exc.value)


def test_partial_failure_prevents_later_agents(model):
    spies = _spies(model)
    class Fails:
        def handle(self, message):
            raise RuntimeError("do not leak private text")
    spies["control"] = Fails()
    coordinator = TravelerOrchestrator(model, agents=spies)
    with pytest.raises(AgentFailure, match="control") as exc:
        coordinator.run("Una reseña")
    assert coordinator.last_execution.state == "DETENIDO_SEGURO"
    assert coordinator.last_execution.trace[-1]["status"] == "FAILED"
    assert coordinator.last_execution.trace[-1]["agent"] == "control"
    assert "Una reseña" not in json.dumps(coordinator.last_execution.trace)
    assert not spies["accion"].messages
    assert "private" not in str(exc.value)


def test_contract_validation_on_malformed_specialist(model):
    spies = _spies(model)
    class Invalid:
        def handle(self, _):
            return {"sentiment": "invented_label"}
    spies["semantica"] = Invalid()
    with pytest.raises(AgentFailure, match="semantica"):
        TravelerOrchestrator(model, agents=spies).run("Una reseña")
    assert not spies["control"].messages


def test_control_requests_human_without_unauthorized_external_action(model):
    spies = _spies(model, threshold=1.01)
    result = TravelerOrchestrator(model, agents=spies).run("La visita fue buena.")
    assert result["revision_humana"] is True
    assert result["tipo_salida"] == "revisión_humana"
    assert result["multiagente"]["estado"] == "REVISION_HUMANA"


def test_specialists_reject_messages_addressed_to_other_tasks(model):
    msg = Message("orquestador", "percepcion", "percepcion", {"text": "Un texto"})
    with pytest.raises(AgentFailure):
        ActionAgent().handle(msg)
    with pytest.raises(AgentFailure):
        ControlAgent(.56).handle(msg)
    with pytest.raises(AgentFailure):
        SemanticsAgent(SimpleNamespace()).handle(msg)
