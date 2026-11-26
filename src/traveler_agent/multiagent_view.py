"""Presentation-only, allowlisted HTML for the real hierarchical execution trace.

No model execution, no agent calls, no transcript or review content.
The UI must NEVER infer completed stages from the fixed plan alone.
"""
from __future__ import annotations

from html import escape
from typing import Any

AGENTS = (
    ("percepcion", "Percepción", "Validación de entrada"),
    ("semantica", "Semántica", "Sentimiento, aspectos, incidentes y evidencias"),
    ("control", "Control", "Continuar o revisión humana"),
    ("accion", "Acción", "Valida el informe y registra la recomendación o abstención"),
)
CLASS_NAMES = {"percepcion": "PerceptionAgent", "semantica": "SemanticsAgent",
               "control": "ControlAgent", "accion": "ActionAgent"}
NAMES = {"orquestador": "Orquestador principal", "qwen": "Qwen local (servicio)",
         **{key: name for key, name, _ in AGENTS}}
TASKS = {
    "plan": "Plan de ejecución", "percepcion": "Percepción",
    "semantica": "Análisis semántico", "control": "Evaluación y control",
    "accion": "Acción y recomendación", "cierre": "Cierre",
    "generacion": "Informe Qwen opcional",
}
STATUSES = {
    "PLANNED": "Planificado", "DELEGATED": "Delegado", "OK": "Completado",
    "FAILED": "Fallido", "COMPLETADO": "Ejecución completada",
    "REVISION_HUMANA": "Revisión humana", "DETENIDO_SEGURO": "Detenido con seguridad",
    "RECEIVED": "Respuesta recibida; pendiente de validación",
}
REASONS = {
    "SCORE_BAJO": "Puntuación inferior al umbral interno",
    "SCORE_DEMO_BAJO": "Puntuación inferior al umbral interno",
    "CONTRASTE_CON_SCORE_LIMITADO": "Contraste léxico con puntuación limitada",
    "TEXTO_TRUNCADO_POR_ENCODER": "Texto truncado por la ventana del encoder",
    "AMBIGUEDAD_O_DUDA": "Ambigüedad o duda detectada en el relato",
    "INCIDENTE_ABIERTO_VS_ETIQUETA": "Incidente abierto frente a etiqueta global",
    "VERIFICACION_QWEN_INCIERTO": "Verificación Qwen: balance incierto",
    "VERIFICACION_QWEN_NO_DISPONIBLE": "Verificación Qwen no disponible",
    "VERIFICACION_QWEN_DISCREPANCIA_NEGATIVO": "Verificación Qwen discrepa (negativo)",
    "VERIFICACION_QWEN_DISCREPANCIA_NEUTRAL": "Verificación Qwen discrepa (neutral)",
    "VERIFICACION_QWEN_DISCREPANCIA_POSITIVO": "Verificación Qwen discrepa (positivo)",
    "VERIFICACION_QWEN_NEGATIVO": "Verificación Qwen discrepa (negativo)",
    "VERIFICACION_QWEN_NEUTRAL": "Verificación Qwen discrepa (neutral)",
    "VERIFICACION_QWEN_POSITIVO": "Verificación Qwen discrepa (positivo)",
}
NOTES = {
    "PLAN_FIJO_CON_GATES": "Plan fijo con controles",
    "HANDOFF": "Delegación",
    "VALIDATED_CONTRACT": "Contrato validado",
    "SAFE_ABORT": "Detención segura",
    "NO_EXTERNAL_ACTION": "Sin acción externa",
    "QWEN_REQUESTED": "Consulta local solicitada tras Control",
    "QWEN_RESPONSE": "Resultado de la consulta local",
    "QWEN_VALIDATED": "Validación del informe por Acción",
}


def _safe(value: Any) -> str:
    return escape(str(value), quote=True)


def _agent_state(events: list[dict], name: str) -> str:
    statuses = [event.get("status") for event in events
                if event.get("agent") == name and event.get("task") == name]
    if "FAILED" in statuses:
        return "FAILED"
    if "OK" in statuses:
        return "OK"
    if any(e.get("agent") == "orquestador" and e.get("task") == name
           and e.get("status") == "DELEGATED" for e in events):
        return "DELEGATED"
    return "NO_EJECUTADO"


def _classifier_block(record: dict) -> str:
    selection = record.get("seleccion_modelo") if isinstance(record.get("seleccion_modelo"), dict) else {}
    multi = record.get("multiagente") if isinstance(record.get("multiagente"), dict) else {}
    runtime = multi.get("clasificador_runtime") if isinstance(multi.get("clasificador_runtime"), dict) else {}
    rows = [
        ("Modelo efectivo", record.get("modelo_utilizado", "pendiente")),
        ("Solicitado", selection.get("requested", "—")),
        ("Fallback", "sí" if selection.get("fallback") is True else "no" if selection.get("fallback") is False else "—"),
        ("Versión", selection.get("version") or selection.get("training_mode") or "—"),
        ("Representación", selection.get("features") or runtime.get("representation") or "—"),
        ("Verificación contextual", "activa" if runtime.get("context_verification") else "inactiva"),
    ]
    body = "".join(f"<div><dt>{_safe(k)}</dt><dd>{_safe(v)}</dd></div>" for k, v in rows)
    return ('<div class="ti-flow-outcome ti-agent-meta"><strong>Clasificador en esta ejecución</strong>'
            f'<dl class="ti-operations">{body}</dl></div>')


def _evidence_block(record: dict) -> str:
    evidence = record.get("fragmentos_literales_no_explicativos") or []
    incidents = record.get("incidentes_exploratorios") or []
    assessment = record.get("valoracion_aspectos_exploratoria") or {}
    parts = []
    if isinstance(assessment, dict) and assessment:
        bits = [f"{_safe(k)}:{_safe(v.get('polarity', '?'))}" for k, v in assessment.items() if isinstance(v, dict)]
        parts.append("<div><strong>Aspectos (exploratorio):</strong> " + _safe("; ".join(bits)) + "</div>")
    if isinstance(incidents, list) and incidents:
        lines = []
        for item in incidents[:4]:
            if not isinstance(item, dict):
                continue
            lines.append(f"{item.get('status', '?')}: «{(item.get('clause') or '')[:120]}»")
        parts.append("<div><strong>Incidentes:</strong> " + _safe(" · ".join(lines)) + "</div>")
    if isinstance(evidence, list) and evidence:
        quotes = " · ".join(f"«{e[:140]}»" for e in evidence[:4] if isinstance(e, str))
        parts.append("<div><strong>Evidencias seleccionadas:</strong> " + _safe(quotes) + "</div>")
    verify = record.get("verificacion_contextual")
    if isinstance(verify, dict):
        parts.append(
            "<div><strong>Verificación Qwen (antes de Control):</strong> "
            f"balance={_safe(verify.get('balance'))}; "
            f"etiqueta preservada={_safe(verify.get('etiqueta_supervisada'))}; "
            f"sobrescribe={_safe(verify.get('sobrescribe_etiqueta'))}</div>"
        )
    if not parts:
        return ""
    return '<div class="ti-flow-outcome ti-agent-meta">' + "".join(parts) + "</div>"


def render_multiagent_view(record: dict | None) -> str:
    multi = record.get("multiagente") if isinstance(record, dict) else None
    if not isinstance(multi, dict) or not isinstance(multi.get("traza"), list):
        return (
            '<section class="ti-flow" aria-label="Coordinación multiagente">'
            '<p class="ti-flow-empty">Sin traza de ejecución disponible. '
            'Si hubo un error, el agente actual no devuelve su traza de fallo '
            'mediante este callback.</p></section>'
        )
    events = [event for event in multi["traza"] if isinstance(event, dict)]
    generation_events = [event for event in multi.get("traza_generacion", []) if isinstance(event, dict)]
    terminal = multi.get("estado")
    status = STATUSES.get(terminal, "Estado no reconocido")
    orchestrator = multi.get("orquestador")
    header = "TravelerOrchestrator" if orchestrator == "TravelerOrchestrator" else "Orquestador sin identificar"
    blocks = [
        '<section class="ti-flow" aria-label="Coordinación multiagente">',
        '<div class="ti-flow-lead">',
        '<div class="ti-flow-eyebrow">ARQUITECTURA MULTIAGENTE · PROTOTIPO ACADÉMICO</div>',
        f'<h3>{_safe(header)}</h3>',
        '<p>Coordina cuatro especialistas mediante delegación y validación de contratos.</p>',
        f'<span class="ti-flow-status">{_safe(status)}</span>',
        '</div>',
        _classifier_block(record),
        _evidence_block(record),
        '<div class="ti-flow-agents" aria-label="Agentes especializados">',
    ]
    for key, label, description in AGENTS:
        raw = _agent_state(events, key)
        display = STATUSES.get(raw, "Sin ejecutar")
        blocks.extend((
            '<div class="ti-flow-agent">',
            f'<strong>{_safe(label)}</strong>',
            f'<div class="ti-agent-class">{CLASS_NAMES[key]}</div>',
            f'<p>{_safe(description)}</p>',
            f'<span class="ti-flow-status">{_safe(display)}</span>',
            '</div>',
        ))
    blocks.extend((
        '</div>',
        render_qwen_service(record),
        '<h4>Secuencia real de eventos</h4>',
        '<ol class="ti-flow-events">',
    ))
    for event in [*events, *generation_events]:
        if generation_events and event is generation_events[0]:
            blocks.append('</ol><h4>Generación · secuencia independiente tras Control</h4>'
                          '<ol class="ti-flow-events">')
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
    if generation_events:
        blocks.append("<p>Generación: secuencia independiente después de Control. "
                      "La verificación contextual, si existe, ocurre antes de Control y no altera la etiqueta.</p>")
    reasons = multi.get("razones_revision")
    if isinstance(reasons, list) and reasons:
        readable = [REASONS.get(reason, reason if isinstance(reason, str) else "Motivo interno")
                    for reason in reasons]
        blocks.append(
            '<div class="ti-flow-outcome"><strong>Motivos de revisión:</strong> '
            + _safe("; ".join(readable)) + "</div>"
        )
    recommendation = record.get("recomendacion")
    if isinstance(recommendation, str) and terminal in {"COMPLETADO", "REVISION_HUMANA"}:
        blocks.append('<div class="ti-flow-outcome"><strong>Recomendación del agente:</strong> '
                      + _safe(recommendation) + "</div>")
    blocks.append(
        '<p class="ti-flow-caveat">Puntuaciones no calibradas externamente. '
        'Planificación fija, memoria episódica de sesión y aprendizaje offline. '
        'Los eventos registran metadatos; aspectos e incidentes son exploratorios.</p></section>'
    )
    return "\n".join(blocks)


def render_qwen_service(record):
    generation = record.get("generacion_qwen") if isinstance(record, dict) else None
    verify = record.get("verificacion_contextual") if isinstance(record, dict) else None
    if not generation:
        status = "Informe: sin consulta"
    elif generation.get("estado") == "GENERADO" and generation.get("validado_por") == "ActionAgent":
        status = "Informe validado por Acción"
    else:
        status = "Informe: sin validar / error"
    verify_line = ""
    if isinstance(verify, dict):
        verify_line = (f'<p>Verificación pre-Control: balance={_safe(verify.get("balance"))}; '
                       f'no sobrescribe etiqueta.</p>')
    return ('<div class="ti-qwen-service"><h4>Qwen / Ollama · servicio generativo subordinado</h4>'
            '<p>Verificación contextual (antes de Control) e informe (después de Control) '
            'están desacoplados. Ninguno ejecuta acciones externas ni reescribe la etiqueta supervisada.</p>'
            + verify_line +
            '<span class="ti-flow-status">' + _safe(status) + '</span></div>')


def execution_map(record):
    multi = record.get("multiagente", {}) if isinstance(record, dict) else {}
    trace = multi.get("traza", []) if isinstance(multi, dict) else []
    language = (record or {}).get("procesamiento_linguistico") or {}
    generation = (record or {}).get("generacion_qwen") or {}
    origin = (record or {}).get("procedencia_entrada") or {}
    verified = (record or {}).get("verificacion_contextual") or {}
    selection = (record or {}).get("seleccion_modelo") or {}
    validated = generation.get("validado_por") == "ActionAgent" and generation.get("estado") == "GENERADO"
    terminal = STATUSES.get(multi.get("estado"), language.get("estado", "Sin ejecutar"))
    human = (record or {}).get("revision_humana")
    asr_state = "Sin audio"
    if origin:
        if origin.get("fue_corregida"):
            asr_state = "Transcripción corregida"
        elif origin.get("revision_percepcion_confirmada"):
            asr_state = "Transcripción confirmada"
        else:
            asr_state = "Transcripción obtenida"
    fields = (
        ("Estado", terminal),
        ("Entrada", multi.get("modalidad_percibida", "Audio" if origin else "Texto" if language else "Sin entrada")),
        ("Audio", asr_state),
        ("Idioma original", language.get("idioma_original", "Sin registro")),
        ("Clasificador", (record or {}).get("modelo_utilizado", "Pendiente")),
        ("Fallback", "sí" if selection.get("fallback") else "no"),
        ("Traducción", "Sin ejecutar" if terminal == "pendiente de revisión ASR" else language.get("estado", "Sin registro")),
        ("Verify Qwen", verified.get("balance", "Sin consulta")),
        ("Informe Qwen", generation.get("estado", "Sin consulta")),
        ("Revisión humana", "Requerida" if human is True else "No requerida" if human is False else "Pendiente"),
        ("Eventos", f'{len(trace)} principales · {len(multi.get("traza_generacion", []))} generativos'),
    )
    summary = "".join(f"<div><dt>{_safe(k)}</dt><dd>{_safe(v)}</dd></div>" for k, v in fields)
    nodes = [("Entrada", "Audio · Whisper" if origin else "Texto" if trace else "Pendiente")]
    for key, label, _ in AGENTS:
        if key == "control":
            nodes.append(("Qwen · verify", verified.get("balance", "Sin consulta") if verified else "Sin consulta"))
        if key == "accion":
            nodes.append(("Qwen · informe", "Validado" if validated else "Error" if generation else "Sin consulta"))
        nodes.append((label, STATUSES.get(_agent_state(trace, key), "Sin ejecutar")))
    nodes.append(("Resultado", terminal))

    def node(label, state):
        tone = "idle" if state in {"Sin ejecutar", "Sin consulta", "Pendiente"} else "error" if state in {"Error", "Fallido"} else "active"
        if state in {"Revisión humana", "pendiente de revisión ASR", "pendiente de revisión"}:
            tone = "review"
        return f'<li class="ti-node-{tone}"><strong>{_safe(label)}</strong><span>{_safe(state)}</span></li>'

    flow = "".join(node(label, state) for label, state in nodes)
    recommendation = (record or {}).get("recomendacion")
    result = f'<p class="ti-decision">{_safe(recommendation)}</p>' if recommendation else ""
    method = language.get("metodo_deteccion", "Sin registro")
    service = language.get("modelo_traduccion") or "Sin traducción automática registrada"
    return (f'<section class="ti-operations"><h2>Resumen de ejecución</h2><dl>{summary}</dl>{result}'
            '<div class="ti-coordinator"><strong>TravelerOrchestrator</strong><span>Coordinación jerárquica</span></div>'
            f'<ol class="ti-runtime-map" aria-label="Flujo de ejecución">{flow}</ol>'
            f'<details><summary>Procedencia lingüística y servicios auxiliares</summary><p>ASR · {_safe(origin.get("motor_asr", "Sin audio"))}'
            f' · {_safe(origin.get("modelo_asr", ""))}</p><p>Identificación · {_safe(method)}</p>'
            f'<p>Traducción · {_safe(service)}</p><p>Audio · {_safe(asr_state)}</p></details></section>')


def render_agents_overview(record=None):
    """Function map when idle; executed states only when a real trace exists."""
    if isinstance(record, dict) and isinstance(record.get("multiagente", {}).get("traza"), list):
        return execution_map(record) + '<details class="ti-timeline"><summary>Cronología, especialistas y contratos</summary>' + render_multiagent_view(record) + "</details>"
    if isinstance(record, dict) and record.get("procesamiento_linguistico"):
        return execution_map(record)
    blocks = ['<section class="ti-flow" aria-label="Coordinación multiagente">',
              '<div class="ti-flow-lead"><div class="ti-flow-eyebrow">ORQUESTADOR PRINCIPAL</div>'
              '<h3>TravelerOrchestrator</h3><p>Funciones del sistema · sin ejecución registrada.</p></div>',
              '<div class="ti-flow-agents">']
    for key, label, description in AGENTS:
        blocks.append('<div class="ti-flow-agent"><strong>' + label + '</strong>'
                      '<div class="ti-agent-class">' + CLASS_NAMES[key] + '</div><p>' + description + '</p>'
                      '<span class="ti-flow-status">Sin ejecutar</span></div>')
    blocks.extend(('</div>', render_qwen_service(None),
                   '<p class="ti-flow-caveat">Analiza una entrada para mostrar la traza real y, si se activa, '
                   'la verificación Qwen previa a Control y el informe posterior.</p></section>'))
    return execution_map({}) + '<details class="ti-timeline"><summary>Funciones de los especialistas</summary>' + "\n".join(blocks) + "</details>"
