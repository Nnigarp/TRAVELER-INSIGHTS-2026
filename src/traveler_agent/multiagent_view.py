"""Presentation-only, allowlisted HTML for the real hierarchical execution trace.

No model execution, no agent calls, no transcript or review content.
The UI must NEVER infer completed stages from the fixed plan alone.
"""
from __future__ import annotations

from html import escape
from typing import Any

AGENTS = (
    ("percepcion", "Percepción", "Validación de entrada"),
    ("semantica", "Semántica", "Sentimiento, aspectos y evidencias"),
    ("control", "Control", "Continuar o revisión humana"),
    ("accion", "Acción", "Recomendación o abstención"),
)
NAMES = {"orquestador": "Orquestador principal", **{key: name for key, name, _ in AGENTS}}
TASKS = {
    "plan": "Plan de ejecución", "percepcion": "Percepción",
    "semantica": "Análisis semántico", "control": "Evaluación y control",
    "accion": "Acción y recomendación", "cierre": "Cierre",
}
STATUSES = {
    "PLANNED": "Planificado", "DELEGATED": "Delegado", "OK": "Completado",
    "FAILED": "Fallido", "COMPLETADO": "Ejecución completada",
    "REVISION_HUMANA": "Revisión humana", "DETENIDO_SEGURO": "Detenido con seguridad",
}
REASONS = {
    "SCORE_DEMO_BAJO": "Puntuación DEMO inferior al umbral interno",
    "CONTRASTE_CON_SCORE_LIMITADO": "Contraste con puntuación DEMO limitada",
}
NOTES = {
    "PLAN_FIJO_CON_GATES": "Plan fijo con controles",
    "HANDOFF": "Delegación",
    "VALIDATED_CONTRACT": "Contrato validado",
    "SAFE_ABORT": "Detención segura",
    "NO_EXTERNAL_ACTION": "Sin acción externa",
}


def _safe(value: Any) -> str:
    return escape(str(value), quote=True)


def _agent_state(events: list[dict], name: str) -> str:
    statuses = [event.get("status") for event in events if event.get("agent") == name]
    if "FAILED" in statuses:
        return "FAILED"
    if "OK" in statuses:
        return "OK"
    # A delegation appears under "orquestador", not under the specialist.
    if any(e.get("agent") == "orquestador" and e.get("task") == name
           and e.get("status") == "DELEGATED" for e in events):
        return "DELEGATED"
    return "NO_EJECUTADO"


def render_multiagent_view(record: dict | None) -> str:
    """Render ONLY recognized trace metadata, no speculative stage status.

    A failure currently raises an exception instead of returning a record.
    Missing trace therefore gets an explicit unavailable message, not a
    fabricated failed event.
    """
    multi = record.get("multiagente") if isinstance(record, dict) else None
    if not isinstance(multi, dict) or not isinstance(multi.get("traza"), list):
        return (
            '<section class="ti-flow" aria-label="Coordinación multiagente">'
            '<p class="ti-flow-empty">Sin traza de ejecución disponible. '
            'Si hubo un error, el agente actual no devuelve su traza de fallo '
            'mediante este callback.</p></section>'
        )
    events = [event for event in multi["traza"] if isinstance(event, dict)]
    # Never alter the captured sequence; display recorded order only.
    # Any extra unknown names/statuses are redacted, not echoed.
    terminal = multi.get("estado")
    status = STATUSES.get(terminal, "Estado no reconocido")
    orchestrator = multi.get("orquestador")
    header = "TravelerOrchestrator" if orchestrator == "TravelerOrchestrator" else "Orquestador sin identificar"
    blocks = [
        '<section class="ti-flow" aria-label="Coordinación multiagente">',
        '<div class="ti-flow-lead">',
        '<div class="ti-flow-eyebrow">ARQUITECTURA MULTIAGENTE · DEMO ACADÉMICA</div>',
        f'<h3>{_safe(header)}</h3>',
        '<p>Coordina cuatro especialistas mediante delegación y validación de contratos.</p>',
        f'<span class="ti-flow-status">{_safe(status)}</span>',
        '</div>',
        '<div class="ti-flow-agents" aria-label="Agentes especializados">',
    ]
    for key, label, description in AGENTS:
        raw = _agent_state(events, key)
        display = STATUSES.get(raw, "Sin ejecutar")
        blocks.extend((
            '<div class="ti-flow-agent">',
            f'<strong>{_safe(label)}</strong>',
            f'<p>{_safe(description)}</p>',
            f'<span class="ti-flow-status">{_safe(display)}</span>',
            '</div>',
        ))
    blocks.extend((
        '</div>',
        '<h4>Secuencia real de eventos</h4>',
        '<ol class="ti-flow-events">',
    ))
    # Only allowlisted task, agent, status and note codes are rendered.
    for event in events:
        agent = NAMES.get(event.get("agent"), "Agente no identificado")
        task = TASKS.get(event.get("task"), "Tarea no identificada")
        state = STATUSES.get(event.get("status"), "Estado no reconocido")
        code = NOTES.get(event.get("note_code"), "")
        number = event.get("sequence")
        number_label = str(number) if type(number) is int and number >= 1 else "?"
        blocks.append(
            '<li class="ti-flow-event">'
            f'<span class="ti-flow-sequence">{_safe(number_label)}</span>'
            '<div>'
            f'<strong>{_safe(agent)}</strong>'
            f'<span class="ti-flow-event-task"> · {_safe(task)}</span>'
            f'<div class="ti-flow-event-note">{_safe(code)}</div>'
            '</div>'
            f'<span class="ti-flow-status">{_safe(state)}</span>'
            '</li>'
        )
    blocks.append("</ol>")
    reasons = multi.get("razones_revision")
    if isinstance(reasons, list) and reasons:
        readable = [REASONS.get(reason, "Motivo interno no reconocido") for reason in reasons]
        blocks.append(
            '<div class="ti-flow-outcome"><strong>Motivos de revisión:</strong> '
            + _safe("; ".join(readable)) + "</div>"
        )
    # A recommendation may be shown only if the original agent produced one.
    recommendation = record.get("recomendacion")
    if isinstance(recommendation, str) and terminal in {"COMPLETADO", "REVISION_HUMANA"}:
        blocks.append('<div class="ti-flow-outcome"><strong>Recomendación del agente:</strong> '
                      + _safe(recommendation) + "</div>")
    blocks.append(
        '<p class="ti-flow-caveat">Los modelos son DEMO; sus puntuaciones no están '
        'calibradas externamente. Planificación fija, memoria temporal y '
        'aprendizaje offline. Los eventos solo registran metadatos, no '
        'razonamientos internos ni transcripciones.</p></section>'
    )
    return "\n".join(blocks)
