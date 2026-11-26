"""Hierarchical *local* multiagent coordination for Traveler Insights.

Agents have distinct observations, contracts, decisions and capabilities.
The coordinator owns a state machine, plan, handoffs, validation and fail-closed
outcomes. Optional Qwen verification may escalate human review but never
overwrites the supervised sentiment label. Optional Ollama reporting cannot
execute external actions. No online learning.

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

from .linguistic import linguistic_profile
from .policy import recommend
from .semantic_context import analyze_context

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
    generation_trace: list[dict] = field(default_factory=list)


class SpecializedAgent(Protocol):
    name: str
    def handle(self, message: Message) -> dict: ...


def _trace(state: Execution, *, agent: str, step: str, status: str,
           input_fields=(), output_fields=(), note: str = "") -> None:
    state.trace.append({
        "sequence": len(state.trace) + 1,
        "agent": agent,
        "task": step,
        "status": status,
        "input_fields": sorted(input_fields),
        "output_fields": sorted(output_fields),
        "note_code": note,
    })


def _trace_generation(state: Execution, *, agent: str, status: str, note: str, generation=None):
    event = {"sequence": len(state.generation_trace) + 1, "agent": agent,
             "task": "generacion", "status": status, "note_code": note,
             "after_task": "control", "before_task": "cierre"}
    if generation is not None:
        event["generation"] = {key: generation.get(key) for key in (
            "estado", "codigo", "modelo_solicitado", "modelo_ejecutado", "latencia_ms", "generacion_llm")}
    state.generation_trace.append(event)


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
            raise AgentFailure("El clasificador no pudo inferir.") from exc
        if (len(classes) != 3 or set(classes) != CLASSES or
            probabilities.shape != (3,) or not np.isfinite(probabilities).all() or
            np.any(probabilities < 0) or
            not np.isclose(probabilities.sum(), 1.0, atol=1e-3)):
            raise AgentFailure("Salida inválida del clasificador; bloquear acción.")
        encoder = getattr(self.classifier, "encoder", None)
        truncated = False
        if encoder is not None:
            try:
                truncated = len(encoder.tokenizer.encode(text, truncation=False)) > encoder.max_length
            except Exception:
                truncated = False
        idx = int(probabilities.argmax())
        from .report_content import is_factual_description
        context = analyze_context(text)
        profile = linguistic_profile(text)
        contrast = bool(re.search(r"\b(pero|aunque|sin embargo)\b", text, re.I))
        return {
            "sentiment": classes[idx],
            "scores": {c: round(float(p), 4) for c, p in zip(classes, probabilities)},
            "confidence_raw": float(probabilities[idx]),
            "aspects": context["aspects"],
            "aspect_assessment": context["aspect_assessment"],
            "incidents": context["incidents"],
            "ambiguity": context["ambiguity"],
            "evidence": context["evidence"] or [],
            "linguistic_profile": profile,
            "factual_description": is_factual_description(text),
            "contrast": contrast,
            "truncated": truncated,
            "context_version": context["version"],
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
        value = semantic["confidence_raw"]
        ambiguity = semantic.get("ambiguity") or {}
        review = value < self.threshold or (semantic["contrast"] and value < .80)
        reasons = []
        if semantic.get("truncated"):
            review = True
            reasons.append("TEXTO_TRUNCADO_POR_ENCODER")
        if ambiguity.get("ambiguous") or ambiguity.get("doubt"):
            review = True
            reasons.append("AMBIGUEDAD_O_DUDA")
        # Open incident under a positive/neutral global label → human check.
        open_incidents = sum(1 for i in (semantic.get("incidents") or []) if i.get("status") == "no_resuelto")
        sentiment = semantic.get("sentiment")
        if semantic.get("factual_description") and sentiment != "neutral":
            review = True
            reasons.append("DESCRIPCION_FACTUAL_VS_ETIQUETA")
        if open_incidents and sentiment in {"positivo", "neutral"}:
            review = True
            reasons.append("INCIDENTE_ABIERTO_VS_ETIQUETA")
        balance = message.payload.get("balance_qwen")
        # Verification never overwrites the supervised label; it only escalates.
        if balance in {"incierto", "no_disponible"}:
            review = True
            reasons.append("VERIFICACION_QWEN_" + balance.upper())
        elif balance is not None and balance in CLASSES and sentiment is not None and balance != sentiment:
            review = True
            reasons.append("VERIFICACION_QWEN_DISCREPANCIA_" + balance.upper())
        if value < self.threshold:
            # Historical reason code kept for acceptance contracts; UI hides the DEMO wording.
            reasons.append("SCORE_DEMO_BAJO")
        if semantic.get("contrast") and value < .80:
            reasons.append("CONTRASTE_CON_SCORE_LIMITADO")
        return {
            "needs_human": review,
            "decision": "SOLICITAR_REVISION" if review else "CONTINUAR",
            "reason_codes": reasons,
            "balance_qwen": balance,
            "supervised_sentiment_preserved": semantic.get("sentiment"),
        }


class ActionAgent:
    name = "accion"

    def handle(self, message: Message) -> dict:
        if message.task != self.name:
            raise AgentFailure("Acción recibió tarea ajena.")
        semantic = message.payload["semantic"]
        control = message.payload["control"]
        context = {
            "incidents": semantic.get("incidents") or [],
            "aspect_assessment": semantic.get("aspect_assessment") or {},
            "ambiguity": semantic.get("ambiguity") or {},
        }
        state, recommendation = recommend(
            semantic["sentiment"], semantic["aspects"], control["needs_human"],
            context=context,
        )
        if control["needs_human"] and state != "revisión_humana":
            raise AgentFailure("Política insegura: revisión humana bloqueada.")
        output = {"policy_state": state, "recommendation": recommendation,
                  "external_actions_executed": False,
                  "incidents": context["incidents"]}
        if "generation" in message.payload:
            from .ollama_report import validate_generation
            output["generation"] = validate_generation(
                message.payload["generation"], message.payload["text"],
                semantic["sentiment"], list(semantic["aspects"]), control["needs_human"])
        return output


class TravelerOrchestrator:
    """Parent agent. Plans, dispatches, validates handoffs and retains status."""

    def __init__(self, classifier: Any, *, threshold: float = .56,
                 agents: dict[str, SpecializedAgent] | None = None,
                 reporter=None, context_verifier=None):
        self.context_verification = getattr(classifier, "context_verification", False)
        self.reporter = reporter
        # Verifier is independent from generative reporting (may share Ollama).
        self.context_verifier = context_verifier
        self.agents = agents if agents is not None else {
            "percepcion": PerceptionAgent(),
            "semantica": SemanticsAgent(classifier),
            "control": ControlAgent(threshold),
            "accion": ActionAgent(),
        }
        if set(self.agents) != set(STEPS):
            raise AgentFailure("El orquestador requiere cuatro especialistas.")
        self.last_execution: Execution | None = None
        self.classifier = classifier

    def run(self, text: str, *, review_id: str = "", modality: str = "texto",
            edited_by_human: bool = False, progress=None) -> dict:
        execution = Execution()
        self.last_execution = execution
        _trace(execution, agent="orquestador", step="plan", status="PLANNED",
               output_fields=STEPS, note="PLAN_FIJO_CON_GATES")
        payload: dict[str, Any] = {
            "text": text, "review_id": review_id,
            "modality": modality, "edited_by_human": edited_by_human,
        }
        execution.state = "EJECUTANDO"
        for step in execution.plan:
            if step == "semantica":
                if progress:
                    progress("Clasificando")
                payload = {"text": execution.results["percepcion"]["text"]}
            elif step == "control":
                semantic = execution.results["semantica"]
                payload = {"semantic": semantic}
                verify = getattr(self.context_verifier, "verify_context", None)
                if self.context_verification and verify and (
                    semantic["contrast"]
                    or semantic.get("ambiguity", {}).get("ambiguous")
                    or re.search(r"\b(no|nunca|sin|tampoco|aunque|pero)\b", text, re.I)
                ):
                    if progress:
                        progress("Verificación contextual")
                    try:
                        payload["balance_qwen"] = verify(text)
                    except Exception:
                        payload["balance_qwen"] = "no_disponible"
            elif step == "accion":
                payload = {
                    "semantic": execution.results["semantica"],
                    "control": execution.results["control"],
                }
                if self.reporter is not None:
                    if progress:
                        progress("Generando informe")
                    from .ollama_report import failed_generation
                    _trace_generation(execution, agent="orquestador", status="DELEGATED", note="QWEN_REQUESTED")
                    try:
                        generation = self.reporter.generate(
                            execution.results["percepcion"]["text"],
                            execution.results["semantica"]["sentiment"],
                            execution.results["control"]["needs_human"],
                            list(execution.results["semantica"]["aspects"]),
                        )
                        if not isinstance(generation, dict):
                            raise ValueError("Invalid report contract")
                    except Exception:
                        generation = failed_generation("GENERADOR_ERROR")
                    _trace_generation(execution, agent="qwen",
                           status="RECEIVED" if generation.get("estado") == "GENERADO" else "FAILED",
                           note="QWEN_RESPONSE", generation=generation)
                    payload.update(generation=generation, text=execution.results["percepcion"]["text"])
                if progress:
                    progress("Validando")
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
                if step == "accion" and self.reporter is not None:
                    from .ollama_report import failed_generation
                    generation = output.get("generation") or failed_generation("INFORME_NO_VALIDADO")
                    if generation.get("estado") == "GENERADO" and generation.get("validado_por") != "ActionAgent":
                        generation = failed_generation("INFORME_NO_VALIDADO")
                    output["generation"] = generation
                    _trace_generation(execution, agent="accion",
                           status="OK" if generation["estado"] == "GENERADO" else "FAILED",
                           note="QWEN_VALIDATED", generation=generation)
            except Exception as exc:
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
        clf_meta = {
            "effective": "contextual" if self.context_verification or getattr(self.classifier, "encoder", None) is not None else "clasico",
            "context_verification": bool(self.context_verification and self.context_verifier),
            "representation": (
                "frozen_beto_mean_pool" if getattr(self.classifier, "artifact_digest", None)
                else "BETO · cabeza supervisada" if getattr(self.classifier, "encoder", None) is not None else "TF-IDF"
            ),
        }
        record = {
            "id": perception["review_id"],
            "sentimiento": semantic["sentiment"],
            "puntuaciones_modelo_no_calibradas": semantic["scores"],
            "confianza_no_calibrada": round(semantic["confidence_raw"], 4),
            "aspectos_exploratorios": list(semantic["aspects"]),
            "valoracion_aspectos_exploratoria": semantic.get("aspect_assessment") or {},
            "incidentes_exploratorios": semantic.get("incidents") or [],
            "ambiguedad_exploratoria": semantic.get("ambiguity") or {},
            "perfil_linguistico_exploratorio": semantic["linguistic_profile"],
            "fragmentos_literales_no_explicativos": semantic["evidence"],
            "revision_humana": control["needs_human"],
            "tipo_salida": action["policy_state"],
            "recomendacion": action["recommendation"],
            "advertencia": (
                "Prototipo académico: las puntuaciones no están calibradas; "
                "los aspectos e incidentes se extraen mediante reglas exploratorias. "
                "No inferir causalidad de los fragmentos. La verificación Qwen no "
                "sobrescribe la etiqueta supervisada."
            ),
            "multiagente": {
                "arquitectura": "jerarquica_local_v1",
                "orquestador": "TravelerOrchestrator",
                "estado": execution.state,
                "plan": list(execution.plan),
                "razones_revision": control["reason_codes"],
                "modalidad_percibida": perception["modality"],
                "transcripcion_editada": perception["edited_by_human"],
                "clasificador_runtime": clf_meta,
                "traza": execution.trace,
                "memoria": "episodica_solo_sesion_sin_aprendizaje_en_linea",
                "acciones_externas": False,
            },
        }
        if control.get("balance_qwen") is not None:
            record["verificacion_contextual"] = {
                "modelo": "qwen3:1.7b",
                "balance": control["balance_qwen"],
                "etapa": "antes_de_control",
                "etiqueta_supervisada": control.get("supervised_sentiment_preserved", semantic["sentiment"]),
                "sobrescribe_etiqueta": False,
                "desacoplada_del_informe": True,
            }
        if self.reporter is not None:
            record["generacion_qwen"] = action["generation"]
            record["multiagente"]["traza_generacion"] = execution.generation_trace
        return record


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
