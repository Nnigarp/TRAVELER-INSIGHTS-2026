"""Hierarchical *local* multiagent coordination for Traveler Insights.

Agents have distinct observations, contracts, decisions and capabilities.
The coordinator owns a state machine, plan, handoffs, validation and fail-closed
outcomes. No LLM, network requests, online learning or external side effects.

Trace is intentionally metadata-only: raw reviews, transcripts and private
audio paths are NOT duplicated into the coordination log.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .aspects import find_aspects, evidence_spans
from .linguistic import linguistic_profile
from .policy import recommend

CLASSES = frozenset({"positivo", "neutral", "negativo"})
STEPS = ("percepcion", "semantica", "control", "accion")
FINAL = frozenset({"COMPLETADO", "REVISION_HUMANA", "DETENIDO_SEGURO"})


class AgentFailure(ValueError):
    """Fail-closed agent exception with a public, non-sensitive reason."""


@dataclass(frozen=True)
class Message:
    """Structured communication contract: task + request + context."""
    sender: str
    recipient: str
    task: str
    payload: dict[str, Any]

    def __post_init__(self):
        if self.task not in STEPS or not isinstance(self.payload, dict):
            raise AgentFailure("Mensaje o tarea no admitida.")


@dataclass
class Execution:
    state: str = "CREADO"
    plan: tuple[str, ...] = STEPS
    results: dict[str, dict] = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)


class SpecializedAgent(Protocol):
    name: str
    def handle(self, message: Message) -> dict: ...


def _trace(state: Execution, *, agent: str, step: str, status: str,
           input_fields=(), output_fields=(), note: str = "") -> None:
    # Metadata only; never save raw text, model outputs or transcript.
    state.trace.append({
        "sequence": len(state.trace) + 1,
        "agent": agent,
        "task": step,
        "status": status,
        "input_fields": sorted(input_fields),
        "output_fields": sorted(output_fields),
        "note_code": note,
    })


class PerceptionAgent:
    name = "percepcion"

    def handle(self, message: Message) -> dict:
        if message.task != self.name:
            raise AgentFailure("Percepción recibió tarea ajena.")
        text = message.payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise AgentFailure("La reseña no puede estar vacía.")
        if len(text) > 5000:
            raise AgentFailure("La reseña excede el límite de 5000 caracteres.")
        modality = message.payload.get("modality", "texto")
        if modality not in ("texto", "audio_transcrito"):
            raise AgentFailure("Modalidad de percepción no admitida.")
        return {
            "text": text.strip(), "modality": modality,
            "review_id": str(message.payload.get("review_id", "")),
            "edited_by_human": bool(message.payload.get("edited_by_human", False)),
        }


class SemanticsAgent:
    name = "semantica"

    def __init__(self, classifier: Any):
        self.classifier = classifier

    def handle(self, message: Message) -> dict:
        if message.task != self.name:
            raise AgentFailure("Semántica recibió tarea ajena.")
        text = message.payload["text"]
        try:
            classes = [str(c) for c in self.classifier.classes_]
            probabilities = np.asarray(self.classifier.predict_proba([text])[0], dtype=float)
        except Exception as exc:
            raise AgentFailure("El clasificador DEMO no pudo inferir.") from exc
        if (len(classes) != 3 or set(classes) != CLASSES or
            probabilities.shape != (3,) or not np.isfinite(probabilities).all() or
            np.any(probabilities < 0) or
            not np.isclose(probabilities.sum(), 1.0, atol=1e-3)):
            raise AgentFailure("Salida inválida del clasificador; bloquear acción.")
        idx = int(probabilities.argmax())
        aspects = find_aspects(text)
        profile = linguistic_profile(text)
        # Retain the 2026 baseline rule: contrast only, with bounded lexical cue.
        contrast = bool(re.search(r"\b(pero|aunque|sin embargo)\b", text, re.I))
        return {
            "sentiment": classes[idx],
            "scores": {c: round(float(p), 4) for c, p in zip(classes, probabilities)},
            "confidence_raw": float(probabilities[idx]),
            "aspects": aspects,
            "evidence": evidence_spans(text, aspects),
            "linguistic_profile": profile,
            "contrast": contrast,
        }


class ControlAgent:
    name = "control"

    def __init__(self, threshold: float):
        if not 0 <= threshold <= 1.01:
            raise AgentFailure("Umbral inválido.")
        self.threshold = threshold

    def handle(self, message: Message) -> dict:
        if message.task != self.name:
            raise AgentFailure("Control recibió tarea ajena.")
        semantic = message.payload["semantic"]
        # The very same threshold/contrast rule as old inspect_review.
        value = semantic["confidence_raw"]
        review = value < self.threshold or (semantic["contrast"] and value < .80)
        reasons = []
        if value < self.threshold:
            reasons.append("SCORE_DEMO_BAJO")
        if semantic["contrast"] and value < .80:
            reasons.append("CONTRASTE_CON_SCORE_LIMITADO")
        return {
            "needs_human": review,
            "decision": "SOLICITAR_REVISION" if review else "CONTINUAR",
            "reason_codes": reasons,
        }


class ActionAgent:
    name = "accion"

    def handle(self, message: Message) -> dict:
        if message.task != self.name:
            raise AgentFailure("Acción recibió tarea ajena.")
        semantic = message.payload["semantic"]
        control = message.payload["control"]
        # Existing policy, no external action or automatic intervention.
        state, recommendation = recommend(
            semantic["sentiment"], semantic["aspects"], control["needs_human"]
        )
        if control["needs_human"] and state != "revisión_humana":
            raise AgentFailure("Política insegura: revisión humana bloqueada.")
        return {"policy_state": state, "recommendation": recommendation,
                "external_actions_executed": False}


class TravelerOrchestrator:
    """Parent agent. Plans, dispatches, validates handoffs and retains status."""

    def __init__(self, classifier: Any, *, threshold: float = .56,
                 agents: dict[str, SpecializedAgent] | None = None):
        self.agents = agents if agents is not None else {
            "percepcion": PerceptionAgent(),
            "semantica": SemanticsAgent(classifier),
            "control": ControlAgent(threshold),
            "accion": ActionAgent(),
        }
        if set(self.agents) != set(STEPS):
            raise AgentFailure("El orquestador requiere cuatro especialistas.")

    def run(self, text: str, *, review_id: str = "", modality: str = "texto",
            edited_by_human: bool = False) -> dict:
        execution = Execution()
        _trace(execution, agent="orquestador", step="plan", status="PLANNED",
               output_fields=STEPS, note="PLAN_FIJO_CON_GATES")
        payload: dict[str, Any] = {
            "text": text, "review_id": review_id,
            "modality": modality, "edited_by_human": edited_by_human,
        }
        execution.state = "EJECUTANDO"
        for step in execution.plan:
            if step == "semantica":
                payload = {"text": execution.results["percepcion"]["text"]}
            elif step == "control":
                payload = {"semantic": execution.results["semantica"]}
            elif step == "accion":
                payload = {
                    "semantic": execution.results["semantica"],
                    "control": execution.results["control"],
                }
            _trace(execution, agent="orquestador", step=step, status="DELEGATED",
                   input_fields=payload.keys(), note="HANDOFF")
            try:
                message = Message("orquestador", step, step, payload)
                output = self.agents[step].handle(message)
                if not isinstance(output, dict):
                    raise AgentFailure("Contrato de respuesta no cumplido.")
                required = {
                    "percepcion": {"text", "review_id", "modality", "edited_by_human"},
                    "semantica": {"sentiment", "scores", "confidence_raw", "aspects",
                                 "evidence", "linguistic_profile", "contrast"},
                    "control": {"needs_human", "decision", "reason_codes"},
                    "accion": {"policy_state", "recommendation", "external_actions_executed"},
                }[step]
                if not required <= set(output):
                    raise AgentFailure("Faltan campos de comunicación del especialista.")
                if step == "semantica" and output["sentiment"] not in CLASSES:
                    raise AgentFailure("Etiqueta de sentimiento no admitida.")
                if step == "control" and (
                    not isinstance(output["needs_human"], bool) or
                    output["decision"] not in {"CONTINUAR", "SOLICITAR_REVISION"} or
                    (output["decision"] == "SOLICITAR_REVISION") != output["needs_human"]
                ):
                    raise AgentFailure("Decisión de control inconsistente.")
                if step == "accion" and (
                    output["external_actions_executed"] is not False or
                    (execution.results["control"]["needs_human"] and
                     output["policy_state"] != "revisión_humana")
                ):
                    raise AgentFailure("Acción externa o revisión eludida.")
                execution.results[step] = output
                _trace(execution, agent=step, step=step, status="OK",
                       input_fields=payload.keys(), output_fields=output.keys(),
                       note="VALIDATED_CONTRACT")
            except Exception as exc:
                # Failure semantics: no recommendation, no partial model output,
                # no leaking text/error details to end-user trace.
                execution.state = "DETENIDO_SEGURO"
                _trace(execution, agent=step, step=step, status="FAILED",
                       note="SAFE_ABORT")
                raise AgentFailure(
                    "Procesamiento detenido de forma segura en " + step + "."
                ) from exc

        perception = execution.results["percepcion"]
        semantic = execution.results["semantica"]
        control = execution.results["control"]
        action = execution.results["accion"]
        execution.state = "REVISION_HUMANA" if control["needs_human"] else "COMPLETADO"
        _trace(execution, agent="orquestador", step="cierre", status=execution.state,
               output_fields=("resultado", "traza"), note="NO_EXTERNAL_ACTION")
        return {
            # UNCHANGED legacy keys and value semantics:
            "id": perception["review_id"],
            "sentimiento": semantic["sentiment"],
            "puntuaciones_modelo_no_calibradas": semantic["scores"],
            "confianza_no_calibrada": round(semantic["confidence_raw"], 4),
            "aspectos_exploratorios": list(semantic["aspects"]),
            "perfil_linguistico_exploratorio": semantic["linguistic_profile"],
            "fragmentos_literales_no_explicativos": semantic["evidence"],
            "revision_humana": control["needs_human"],
            "tipo_salida": action["policy_state"],
            "recomendacion": action["recommendation"],
            "advertencia": (
                "Prototipo académico: las puntuaciones no están calibradas; "
                "los aspectos se extraen mediante reglas. "
                "No inferir causalidad de los fragmentos."
            ),
            # New optional metadata; existing JSON, UI and exports still work.
            "multiagente": {
                "arquitectura": "jerarquica_local_v1",
                "orquestador": "TravelerOrchestrator",
                "estado": execution.state,
                "plan": list(execution.plan),
                "razones_revision": control["reason_codes"],
                "modalidad_percibida": perception["modality"],
                "transcripcion_editada": perception["edited_by_human"],
                "traza": execution.trace,
                "memoria": "episodica_solo_sesion_sin_aprendizaje_en_linea",
                "acciones_externas": False,
            },
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Caso demostrativo multiagente local.")
    parser.add_argument("--text", default="El guía fue amable, pero esperamos demasiado para entrar.")
    parser.add_argument("--model", choices=("lr", "svm"), default="lr")
    args = parser.parse_args(argv)
    from .demo_data import DEMO
    from .model import train
    clf = train([x for x, _ in DEMO], [y for _, y in DEMO], model_kind=args.model)
    record = TravelerOrchestrator(clf).run(args.text)
    record["modelo_utilizado"] = args.model
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
