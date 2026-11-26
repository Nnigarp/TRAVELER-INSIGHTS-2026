"""Read-only HTML views of actual backend records. No predictions or actions."""
from html import escape
import re

BRAND_HTML = '''<div class="ti-brand"><div class="ti-brand-mark" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M3 17 8 7l4 7 3-4 6 8" stroke-linecap="round" stroke-linejoin="round"/><circle cx="18" cy="5" r="2" fill="#fcd34d" stroke="none"/></svg></div><div><div class="ti-brand-name">Traveler Insights <span>2026</span></div><div class="ti-brand-sub">CARTAGENA · UTB INTELLIGENCE</div></div></div>'''
HERO_HTML = '''<header class="ti-hero"><h1>¿Qué experiencia turística deseas analizar?</h1>
<p>Escribe una reseña, dicta tu experiencia o adjunta un audio.</p></header>'''
AGENT_ORDER_HTML = '''<ol class="ti-flow-order" aria-label="Orden de las etapas coordinadas por TravelerOrchestrator">
<li>Entrada</li><li>Percepción</li><li>Semántica</li><li>Control</li>
<li class="ti-flow-optional">Qwen · opcional</li><li>Acción</li><li>Resultado</li></ol>'''
EMPTY_REPORT_HTML = '''<div class="ti-placeholder ti-result-empty"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><path d="M6 3h9l4 4v14H5V3h1zm9 0v5h4M8 12h8M8 16h6"/></svg><h3>Tu informe aparecerá aquí</h3><p>Escribe una reseña o carga un ejemplo demostrativo y pulsa Analizar reseña.</p></div>'''
EMPTY_AUDIO_REPORT_HTML = EMPTY_REPORT_HTML.replace(
    "Escribe una reseña o carga un ejemplo demostrativo y pulsa Analizar reseña.",
    "Transcribe un audio, revisa el texto y pulsa Analizar transcripción corregida.")


def safe_text(value):
    text = escape(str(value), quote=True)
    text = re.sub(r"([\\`*_{}\[\]()!#|])", lambda m: f"&#{ord(m[1])};", text)
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def section_heading(eyebrow, title, description):
    return ('<div class="ti-section-heading"><div class="ti-eyebrow">' + safe_text(eyebrow) +
            '</div><h1>' + safe_text(title) + '</h1><p class="ti-copy">' + safe_text(description) + '</p></div>')


def pane_heading(title, description):
    return '<div class="ti-pane-head"><h2>' + safe_text(title) + '</h2><p>' + safe_text(description) + '</p></div>'


def render_result(record, *, model_kind=None, audio=False):
    """Never manufacture a generated report from conventional fields."""
    if not isinstance(record, dict) or not record:
        return EMPTY_REPORT_HTML
    generation = record.get("generacion_qwen") or {}
    generated = (generation.get("estado") == "GENERADO" and generation.get("generacion_llm") is True
                 and bool(generation.get("informe")))
    report = generation.get("informe") if generated else {}
    effective = record.get("modelo_utilizado", model_kind) or "No registrado"
    sentiment = record.get("sentimiento", report.get("sentimiento_analizado", "No registrado"))
    aspects = record.get("aspectos_exploratorios", report.get("aspectos_identificados", []))
    review = record.get("revision_humana", report.get("requiere_revision_humana"))
    def line(label, value):
        return '<p><strong>' + label + ':</strong> ' + safe_text(value) + '</p>'
    def block(label, value, variant=""):
        return '<div class="ti-report-block ' + variant + '"><h4>' + label + '</h4><p>' + safe_text(value) + '</p></div>'
    def indicator(label, value, variant=""):
        return '<span class="ti-indicator ' + variant + '">' + label + '<strong>' + safe_text(value) + '</strong></span>'
    parts = ['<section class="ti-report" aria-label="Informe turístico">']
    parts.append('<p class="ti-report-source" role="status">' + (
        'Generado con Qwen y validado por Acción · propuestas sin ejecutar'
        if generated else 'Salida tradicional basada en reglas · sin informe Qwen validado') + '</p>')
    if generated:
        mode = generation.get("validacion_contenido", {}).get("resumen")
        title = "Resumen ejecutivo" + (" · cita conservadora" if mode == "CITA_CONSERVADORA" else "")
        summary = report["resumen"]
    else:
        title = "Resultado del análisis"
        summary = ("La reseña queda pendiente de revisión humana."
                   if review else "El clasificador completó el análisis sin activar la revisión humana.")
    parts.append('<div class="ti-report-summary"><h4>' + title + '</h4><p>' + safe_text(summary) + '</p></div>')
    generation_state = ("Generado y validado" if generated else
                        "Error · " + str(generation.get("codigo", "SIN_INFORME")) if generation else "Desactivado")
    selection = record.get("seleccion_modelo") or {}
    if str(effective).lower() == "beto":
        sentiment_label = "Sentimiento (BETO)"
    elif selection.get("version") == "V4.3" or selection.get("features") == "frozen_beto_mean_pool":
        sentiment_label = "Sentimiento (V4.3 contextual)"
    else:
        sentiment_label = "Sentimiento (clasificador local)"
    variant = "ti-sent-" + sentiment if sentiment in {"positivo", "neutral", "negativo"} else ""
    verify = record.get("verificacion_contextual") or {}
    verify_label = verify.get("balance", "Sin consulta") if verify else "Sin consulta"
    parts.extend(('<div class="ti-indicators" aria-label="Indicadores del análisis">',
                  indicator(sentiment_label, sentiment, variant), indicator("Clasificador", str(effective).upper()),
                  indicator("Aspectos exploratorios", ", ".join(aspects) or "No identificados"),
                  indicator("Revisión humana", "Requerida" if review else "No activada"),
                  indicator("Verify Qwen", str(verify_label)),
                  indicator("Generación Qwen", generation_state), '</div>'))
    parts.append('<div class="ti-report-grid">')
    language = record.get("procesamiento_linguistico", {})
    if language:
        parts.append(line("Idioma original", language["idioma_original"]))
        parts.append(line("Traducción", language["estado"]))
    if generated:
        problem = report["problema_identificado"] or "Incidente no determinado por el contrato; revisar la evidencia."
        recommendation, action = report["recomendacion"], report["accion_propuesta"]
        evidence = report.get("evidencia_textual", [])
    else:
        incidents = record.get("incidentes_exploratorios") or []
        open_inc = [i for i in incidents if isinstance(i, dict) and i.get("status") == "no_resuelto"]
        if open_inc:
            problem = "Incidente exploratorio (reglas): " + (open_inc[0].get("clause") or "mencionado")
        elif incidents:
            problem = "Incidentes exploratorios registrados; revisar evidencia. Sin informe generativo."
        else:
            problem = "Sin incidente explícito detectado por reglas. El análisis conserva sentimiento, aspectos y Control."
        evidence = record.get("fragmentos_literales_no_explicativos", [])
        recommendation = record.get("recomendacion", "Sin recomendación registrada.")
        action = "No se ha generado una acción adicional. La recomendación por reglas requiere valoración humana."
    parts.append(block("Problema identificado", problem, "ti-report-problem"))
    parts.append('<div class="ti-report-block ti-report-evidence"><h4>Evidencia textual</h4>')
    if language.get("traduccion_automatica") or language.get("traduccion_editada"):
        parts.append("<p>Citas de la traducción al español.</p>")
    if evidence:
        parts.extend('<blockquote>' + safe_text(quote) + '</blockquote>' for quote in evidence)
    else:
        parts.append('<p>' + ('Protocolo anterior sin evidencia separada.' if generated else
                            'No se extrajeron fragmentos literales mediante las reglas.') + '</p>')
    parts.append('</div>')
    parts.append(block("Recomendación contextual" if generated else "Recomendación exploratoria", recommendation, "ti-report-rec"))
    parts.append(block("Acción propuesta (sin ejecutar)", action, "ti-report-action"))
    parts.append('</div>')
    warnings = [record.get("advertencia", "")]
    if generated:
        warnings.extend(report.get("advertencias", []))
    elif generation:
        warnings.append("Qwen no produjo un informe validado. Se conserva el análisis convencional y su recomendación por reglas; generacion_llm=false.")
    if str(effective).lower() == "beto":
        warnings.append("BETO experimental sintético usa una ventana limitada de tokens; revisa el contexto completo. Su política no valida automatizar decisiones.")
    elif selection.get("version") == "V4.3" or selection.get("features") == "frozen_beto_mean_pool":
        warnings.append("Clasificador contextual V4.3 (embeddings BETO congelados + cabeza clásica); etiquetas de entrenamiento sintéticas, sin validación humana externa.")
    else:
        warnings.append("Clasificador local con ejemplos sintéticos de demostración; sin validación predictiva turística externa.")
    if selection.get("fallback"):
        warnings.append("BETO no completó el análisis; se ejecutó LR de respaldo explícito.")
    if verify:
        warnings.append("La verificación Qwen no sobrescribe la etiqueta supervisada; solo puede solicitar revisión humana.")
    if record.get("etiqueta_corregida_humano") is not None:
        warnings.append("Informe anterior a la corrección humana; no se ha regenerado. Revisa las propuestas junto con la etiqueta corregida.")
    parts.append('<aside class="ti-warnings"><h4>Advertencias</h4>')
    parts.append('<p>' + safe_text(" ".join(w for w in warnings if w)) + '</p>')
    parts.append(line("Revisión humana", "Requerida por ControlAgent." if review else
                      "No activada por ControlAgent; no acredita automatización segura."))
    parts.append('<p>Ninguna acción externa se ha ejecutado.</p></aside>')
    parts.append('<details class="ti-report-tech"><summary>Información técnica de generación</summary>')
    parts.extend((line("Modelo ejecutado", effective), line("Estado Ollama/generación", generation_state),
                  line("generacion_llm", bool(generated)), line("Modelo solicitado", generation.get("modelo_solicitado", "No solicitado")),
                  line("Modelo que respondió", generation.get("modelo_ejecutado") or "Sin respuesta verificada"),
                  line("Protocolo", generation.get("protocolo", "No solicitado")),
                  line("Tiempo de respuesta (ms)", generation.get("latencia_ms", "No registrado")),
                  line("Redacción", generation.get("validacion_contenido", {}).get("resumen", "No registrada"))))
    if record.get("puntuaciones_modelo_no_calibradas"):
        parts.append("<h4>Puntuaciones no calibradas</h4>")
        for label, value in record["puntuaciones_modelo_no_calibradas"].items():
            parts.append(f'<p>{safe_text(label)} <meter min="0" max="1" value="{float(value)}"></meter> {float(value):.4f}</p>')
    if audio and record.get("procedencia_entrada"):
        parts.append(line("Transcripción corregida", "Sí" if record["procedencia_entrada"].get("fue_corregida") else "No"))
    parts.append('<p>El JSON completo y la trazabilidad conservan los datos originales, sin modificaciones de presentación.</p></details></section>')
    return "\n".join(parts)


def render_generation_html(record):
    """Compatibility entrypoint for historical tests and batch report cards."""
    return render_result(record) if isinstance(record, dict) and record.get("generacion_qwen") else ""
