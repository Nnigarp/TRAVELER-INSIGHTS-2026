"""Local demo UI: individual/batch analysis with auditable human corrections.

No external sharing, scientific metrics, or automatic online learning.
"""
from __future__ import annotations

import json
from functools import lru_cache
import tempfile
from pathlib import Path
import gradio as gr
from traveler_agent.language_input import prepare, check_batch_languages
from traveler_agent.agent import inspect_review
from traveler_agent.multiagent_view import render_multiagent_view, render_agents_overview
from traveler_agent.audio_asr import AudioInputError, recognize_audio, analyze_edited_transcript
from traveler_agent.batch import (
    analyze_batch, apply_human_correction, demo_trends, export_csv,
    export_json, parse_review_csv, parse_review_lines,
)
from traveler_agent.demo_data import DEMO
from traveler_agent.model import MODEL_KINDS, load_model, save_model, train
from traveler_agent.model_runtime import configured_artifact, run_selected
from traveler_agent.beto_distribution import artifact_available
from traveler_agent.ollama_report import render_generation_html

MODEL_PATH = Path("artifacts/demo_model.joblib")
if not MODEL_PATH.exists():
    model = train([t for t, _ in DEMO], [y for _, y in DEMO])
    save_model(model, MODEL_PATH, provenance="SINTETICO_DEMO_NO_PUBLICABLE")
model = load_model(MODEL_PATH)

@lru_cache(maxsize=2)
def get_demo_model(kind: str = "lr"):
    if kind not in MODEL_KINDS:
        raise ValueError("Modelo no disponible")
    if kind == "lr":
        return model
    # Local, temporary SVM demo: no download, no scientific benchmark.
    return train([t for t, _ in DEMO], [y for _, y in DEMO], model_kind="svm")



from traveler_agent.ui_style import THEME, UI_CSS, THEME_JS, INITIAL_THEME_JS, THEME_CHOICES, _PRIMARY
from traveler_agent.ui_presenter import (
    BRAND_HTML, HERO_HTML, AGENT_ORDER_HTML, EMPTY_REPORT_HTML, EMPTY_AUDIO_REPORT_HTML,
    render_result, section_heading, pane_heading, safe_text,
)


def _analysis_card(record: dict, *, model_kind: str, audio: bool = False) -> str:
    return render_result(record, model_kind=model_kind, audio=audio)


def shares_transcript(text, original):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, original.casefold(), text.casefold(), autojunk=False).ratio() >= 0.45


def unified(review, model_kind, qwen_enabled, origin=None, language="", edited="", prior=None, progress=lambda stage: None):
    trace = {}
    try:
        if not isinstance(review, str) or not review.strip() or len(review) > 5000:
            raise ValueError("Introduce una reseña de 1 a 5000 caracteres.")
        trace = prepare(review, asr=(origin or {}).get("idioma_detectado_asr"), manual=language,
                        edited=edited, prior=prior, progress=progress)
        if not trace["texto_analizado"]:
            raise ValueError(" ".join(trace["advertencias"]) + " Revisa la traducción y vuelve a analizar.")
        target = trace["texto_analizado"]
        corrected = bool(origin and origin["transcripcion_original"] != review)
        record, effective = run_selected(model_kind, get_demo_model,
            lambda classifier: inspect_review(target, classifier, modality="audio_transcrito" if origin else "texto",
                                              edited_by_human=corrected, qwen_enabled=qwen_enabled, progress=progress))
        record["modelo_utilizado"] = effective
        record["procesamiento_linguistico"] = trace
        if origin:
            record["procedencia_entrada"] = {**origin, "transcripcion_editada": review, "fue_corregida": corrected, "texto_analizado": target}
        progress("Completado")
        return _analysis_card(record, model_kind=effective, audio=bool(origin)), json.dumps(record, ensure_ascii=False, indent=2), trace
    except (ValueError, TypeError) as exc:
        return '<p role="alert">' + safe_text(f"Error recuperable: {exc}") + '</p>', "{}", trace


def analyze(review: str, model_kind: str = "lr", qwen_enabled: bool = False):
    return unified(review, model_kind, qwen_enabled)[:2]


def analyze_with_agents(review: str, model_kind: str = "lr", qwen_enabled: bool = False, language: str = ""):
    """UI-only adapter: exactly one inference, unchanged legacy JSON outputs.

    A safe failure currently has no returned execution trace. Do not fabricate
    specialist status or a completed run when the underlying agent raised.
    """
    if language:
        summary, json_text, _ = unified(review, model_kind, qwen_enabled, language=language)
    else:
        summary, json_text = analyze(review, model_kind, True) if qwen_enabled else analyze(review, model_kind)
    try:
        record = json.loads(json_text)
    except (TypeError, json.JSONDecodeError):
        record = {}
    return summary, json_text, render_multiagent_view(record)


def _render_batch(records: list[dict]):
    table = [[r["id"], r["texto_original"], r.get("modelo_utilizado", "lr"), r["sentimiento"],
              r["etiqueta_corregida_humano"] or "", r["estado_revision"]]
             for r in records]
    counts = [[x["sentimiento_modelo"], x["cantidad"]] for x in demo_trends(records)]
    ids = [r["id"] for r in records]
    return records, table, counts, gr.update(choices=ids, value=ids[0] if ids else None)


def _generation_batch_status(records):
    generations = [r["generacion_qwen"] for r in records if "generacion_qwen" in r]
    if not generations:
        return ""
    passed = sum(g.get("estado") == "GENERADO" for g in generations)
    return f" Qwen: {passed}/{len(generations)} informes validados; {len(generations)-passed} errores explícitos."


def batch_generation_view(records):
    return "\n".join(render_generation_html(record) for record in records or [])


def run_batch_lines(raw: str, model_kind: str = "lr", qwen_enabled: bool = False):
    try:
        reviews = parse_review_lines(raw)
        check_batch_languages(reviews)
        records, effective = run_selected(model_kind, get_demo_model,
            lambda classifier: analyze_batch(reviews, classifier, model_kind=model_kind, qwen_enabled=qwen_enabled))
        status = "Lote de demostración analizado."
        if model_kind == "beto": status += " BETO experimental." if effective == "beto" else " BETO falló; se ejecutó LR de respaldo."
        return _render_batch(records) + (status + _generation_batch_status(records),)
    except ValueError as exc:
        return _render_batch([]) + (f"Error de validación: {exc}",)


def run_batch_csv(filepath: str | None, model_kind: str = "lr", qwen_enabled: bool = False):
    if not filepath:
        return _render_batch([]) + ("Selecciona un CSV UTF-8 con columna 'reseña'.",)
    try:
        path = Path(filepath)
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("El archivo supera el límite de 1 MiB.")
        reviews = parse_review_csv(path.read_bytes())
        check_batch_languages(reviews)
        records, effective = run_selected(model_kind, get_demo_model,
            lambda classifier: analyze_batch(reviews, classifier, model_kind=model_kind, qwen_enabled=qwen_enabled))
        status = "CSV de demostración analizado."
        if model_kind == "beto": status += " BETO experimental." if effective == "beto" else " BETO falló; se ejecutó LR de respaldo."
        return _render_batch(records) + (status + _generation_batch_status(records),)
    except (OSError, ValueError) as exc:
        return _render_batch([]) + (f"Error de validación: {exc}",)


def revise(records: list[dict], review_id: str, label: str, note: str):
    try:
        edited = apply_human_correction(records, review_id, label, note)
        return _render_batch(edited) + ("Corrección humana registrada en la sesión (sin modificar el modelo).",)
    except ValueError as exc:
        return _render_batch(records or []) + (f"Error de validación: {exc}",)


def export_files(records: list[dict]):
    if not records:
        return None, None, "Analiza un lote antes de exportar."
    directory = Path(tempfile.mkdtemp(prefix="traveler_demo_"))
    json_path = directory / "analisis_demo.json"
    csv_path = directory / "analisis_demo.csv"
    json_path.write_text(export_json(records), encoding="utf-8")
    # csv.writer already chooses record separators. Prevent Windows text mode
    # from turning a quoted CRLF into CR-CR-LF and changing the original review.
    csv_path.write_text(export_csv(records), encoding="utf-8-sig", newline="")
    return str(json_path), str(csv_path), (
        "Archivos DEMO generados localmente. Pueden contener el texto ingresado; trátalos como privados."
    )




def transcribe_audio_ui(filepath: str | None, language: str):
    """Returns editable transcript, metadata state and human-readable status."""
    try:
        import hashlib
        from gradio import processing_utils
        if not filepath:
            raise AudioInputError("Selecciona o graba un audio.")
        path = Path(filepath)
        if path.stat().st_size > 12 * 1024 * 1024:
            raise AudioInputError("El audio supera 12 MiB.")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.suffix.lower() == ".wav":
            text, provenance = recognize_audio(filepath, language)
        else:
            with tempfile.TemporaryDirectory() as directory:
                wav = str(Path(directory) / "asr.wav")
                try:
                    rate, samples = processing_utils.audio_from_file(str(path))
                    processing_utils.audio_to_file(rate, samples, wav, format="wav")
                except Exception as exc:
                    raise AudioInputError("No se pudo convertir el archivo a WAV.") from exc
                text, provenance = recognize_audio(wav, language)
        provenance["audio_original_sha256"] = digest
        return text, provenance, (
            f"Transcripción local terminada ({provenance['duracion_segundos']} s; "
            f"idioma ASR: {provenance['idioma_detectado_asr']}). "
            "Revisa especialmente negaciones, nombres y ruido antes de analizar."
        )
    except (AudioInputError, OSError) as exc:
        return "", {}, f"Error de audio: {exc}"


def analyze_audio_ui(edited_text: str, provenance: dict, model_kind: str, qwen_enabled: bool = False):
    try:
        result, effective = run_selected(model_kind, get_demo_model,
            lambda classifier: analyze_edited_transcript(edited_text, provenance, classifier,
                                                        model_kind=model_kind, qwen_enabled=qwen_enabled))
        summary = _analysis_card(result, model_kind=effective, audio=True)
        return summary, json.dumps(result, ensure_ascii=False, indent=2)
    except (AudioInputError, ValueError, TypeError) as exc:
        return '<p role="alert">' + safe_text(f"Error: {exc}") + '</p>', "{}"


def reset_audio_trace():
    """Do not analyze stale transcripts after replacing a recording."""
    return "", {}, "", "", "{}"

# UI-only helpers: never infer or alter captured backend records.
DEMO_EXAMPLES = {
    "Mixta": "El hotel era cómodo, pero el transporte nunca llegó y perdimos la excursión.",
    "Positiva": "El guía fue amable. La habitación estaba limpia.",
    "Ambigua": "No sé si volvería.",
}


def character_count(text):
    count = len(text or "")
    return f"{count:,} / 5.000 caracteres" + (" · límite excedido" if count > 5000 else "")


def capture_execution(encoded, source="Análisis individual"):
    try:
        record = json.loads(encoded)
        if not isinstance(record, dict) or not record.get("multiagente", {}).get("traza"):
            record = None
    except (ValueError, TypeError, AttributeError):
        record = None
    return (record, render_agents_overview(record), encoded if record else "{}",
            source if record else "Sin ejecución registrada en esta sesión.")


def capture_batch_execution(records, review_id):
    selected = next((r for r in records or [] if r.get("id") == review_id), None)
    return capture_execution(json.dumps(selected, ensure_ascii=False), "Lote · " + str(review_id))


def capture_audio_execution(encoded):
    return capture_execution(encoded, "Audio · transcripción revisada")


def clear_analysis():
    return ("", character_count(""), EMPTY_REPORT_HTML, "{}", render_multiagent_view(None),
            *capture_execution("{}"))


def load_example(name):
    values = list(clear_analysis())
    values[0] = DEMO_EXAMPLES[name]
    values[1] = character_count(values[0])
    return tuple(values)


def configuration_context(model_kind, qwen_enabled):
    names = {"lr": "Regresión logística", "svm": "SVM lineal", "beto": "BETO V3"}
    model_name = names.get(model_kind, "Sin selección")
    generation = "solicitado para el próximo análisis" if qwen_enabled else "desactivado"
    return f"**Configuración de Análisis** · {model_name} · Qwen {generation}"


# Gradio 6 accepts launch-time theme/css; 4–5 attach them to Blocks.
_GRADIO_MAJOR = int(gr.__version__.split(".")[0])
_BLOCKS_THEME = {"theme": THEME, "css": UI_CSS} if _GRADIO_MAJOR < 6 else {}
_BETO_AVAILABLE = artifact_available(configured_artifact())
with gr.Blocks(title="Traveler Insights 2026 | Cartagena · UTB", analytics_enabled=False, fill_width=True, **_BLOCKS_THEME) as app:
    last_record = gr.State(None)
    with gr.Row(elem_id="ti-topbar"):
        with gr.Column(scale=3, min_width=260):
            gr.HTML(BRAND_HTML)
        with gr.Column(scale=0, min_width=372):
            theme_choice = gr.Radio(list(THEME_CHOICES), value=THEME_CHOICES[0],
                                    label="Tema visual", elem_id="ti-theme")
    with gr.Tabs(selected="analysis", elem_id="ti-nav") as navigation:
        with gr.Tab("Análisis", id="analysis", elem_id="ti-analysis-tab"):
            gr.HTML(HERO_HTML)
            with gr.Row(elem_id="ti-analysis-row"):
                with gr.Column(scale=4, min_width=300, elem_id="ti-input-pane", elem_classes=["ti-pane"]):
                    gr.HTML(pane_heading("Nueva evaluación", "Configura los modelos y analiza una reseña turística."))
                    _MODEL_CHOICES = [("TF-IDF · Regresión logística", "lr"), ("TF-IDF · SVM lineal", "svm")]
                    if _BETO_AVAILABLE:
                        _MODEL_CHOICES.append(("BETO V3 · Transformer experimental", "beto"))
                    model_choice = gr.Dropdown(label="Modelo de demostración", choices=_MODEL_CHOICES, value="lr", elem_id="ti-model")
                    qwen_choice = gr.Checkbox(label="Activar informe local Qwen3 1.7B (Ollama)", value=False,
                                             info="La traducción automática usa Qwen aunque el informe esté desactivado.", elem_id="ti-qwen")
                    inp = gr.Textbox(label="Reseña turística", lines=6, value="",
                                     placeholder="Escribe o dicta tu experiencia en tu idioma…", elem_id="ti-review")
                    char_count = gr.Markdown(character_count(""), elem_id="ti-char-count")
                    with gr.Accordion("+ Audio · micrófono o archivo", open=False):
                        audio_input = gr.Audio(label="Grabar o adjuntar audio", sources=["microphone", "upload"], type="filepath", format=None, elem_id="ti-audio-input")
                        audio_language = gr.Dropdown(label="Idioma de Whisper", choices=["auto", "es", "en", "it"], value="auto")
                        audio_status = gr.Markdown("Whisper Tiny · revisa la transcripción antes de analizar.")
                        audio_metadata = gr.State({})
                        pending_audio = gr.State(None)
                        discard_audio = gr.Button("Descartar audio", size="sm")
                        gr.Markdown("Para empezar una reseña independiente, pulsa Limpiar.")
                    with gr.Accordion("Idioma y traducción experimental", open=False):
                        manual = gr.Textbox(value="", label="Idioma original (opcional, código ISO)")
                        translation = gr.Textbox(value="", label="Traducción al español revisada", lines=3)
                        confirmed = gr.Checkbox(label="He revisado la traducción señalada", value=False)
                        previous = gr.State({})
                        gr.Markdown("Si aparece una advertencia, comprueba la traducción antes de volver a pulsar Analizar reseña.")
                    api_run = gr.Button(visible=False)
                    gr.HTML('<p class="ti-example-note">Ejemplos de reseñas · cargan texto para editar antes de analizar.</p>')
                    with gr.Row(elem_classes=["ti-example-row"]):
                        example_mixed = gr.Button("Mixta · transporte", size="sm")
                        example_positive = gr.Button("Positiva · servicio", size="sm")
                        example_uncertain = gr.Button("Ambigua · regreso", size="sm")
                    with gr.Row():
                        run = gr.Button("Analizar reseña", variant="primary", elem_id="ti-analyze")
                        clear = gr.Button("Limpiar", elem_id="ti-clear")
                    cancel = gr.Button("Cancelar análisis", visible=False, elem_id="ti-cancel")
                    gr.HTML('<div class="ti-input-note"><strong>Inferencia local real.</strong> LR/SVM usan datos ficticios y BETO es experimental. Qwen no cambia clasificación ni revisión humana. No se ejecutan acciones externas.</div>')
                with gr.Column(scale=6, min_width=300, elem_id="ti-result-pane", elem_classes=["ti-pane"]):
                    gr.HTML(pane_heading("Informe de experiencia", "Sentimiento, evidencia y revisión de la reseña."))
                    result = gr.HTML(EMPTY_REPORT_HTML, elem_id="ti-result", label="Informe de experiencia")
                    with gr.Accordion("Detalles técnicos · JSON y trazabilidad", open=False, elem_classes=["ti-technical"]):
                        detail = gr.Code(value="{}", label="JSON completo sin modificaciones", language="json", max_lines=18, wrap_lines=True, interactive=False, elem_id="ti-detail-json")
                        agent_view = gr.HTML(render_multiagent_view(None), label="Traza de esta ejecución")
                    goto_agents = gr.Button("Explorar trazabilidad", size="sm")
        with gr.Tab("Lotes", id="batch", elem_id="ti-batch-tab"):
            gr.HTML(section_heading("Revisión de varias experiencias", "Análisis por lotes", "Procesa texto o CSV, conserva predicciones y correcciones y exporta resultados reales. Los conteos son descriptivos."))
            with gr.Row(elem_id="ti-batch-config"):
                batch_config = gr.Markdown(configuration_context("lr", False), elem_classes=["ti-config-context"])
                batch_settings = gr.Button("Cambiar configuración", size="sm", scale=0)
            batch_state = gr.State([])
            with gr.Row():
                with gr.Column(min_width=300, elem_classes=["ti-pane"]):
                    gr.HTML(pane_heading("Reseñas del lote", "Máximo 200 reseñas · CSV UTF-8 de hasta 1 MiB con columna reseña"))
                    batch_text = gr.Textbox(label="Reseñas (una por línea)", lines=7, placeholder="Una reseña por línea. No hay resultados hasta analizar.", elem_id="ti-batch-text")
                    run_lines = gr.Button("Analizar texto por lotes", variant="primary")
                    batch_file = gr.File(label="Cargar CSV local", file_types=[".csv"], type="filepath", height=140, elem_id="ti-batch-file")
                    run_csv = gr.Button("Analizar CSV")
                    gr.Markdown("Lotes admite reseñas identificadas localmente como españolas. Para otros idiomas usa Análisis. El informe Qwen consulta hasta cinco reseñas en secuencia con presupuesto compartido. El resto conserva análisis y recomendación por reglas.")
                with gr.Column(min_width=300, elem_classes=["ti-pane"]):
                    gr.HTML(pane_heading("Revisión humana", "La corrección se registra sin cambiar ni reentrenar el modelo."))
                    review_id = gr.Dropdown(label="ID de reseña", choices=[])
                    corrected = gr.Dropdown(label="Etiqueta revisada", choices=["positivo", "neutral", "negativo"], value="neutral", elem_id="ti-correction")
                    note = gr.Textbox(label="Motivo opcional (máximo 500 caracteres)")
                    save_review = gr.Button("Registrar corrección", variant="primary")
                    batch_goto_agents = gr.Button("Ver traza de la reseña seleccionada")
                    gr.HTML(pane_heading("Exportaciones", "Descarga los resultados y las correcciones del lote."))
                    make_export = gr.Button("Preparar exportaciones")
                    with gr.Row():
                        json_export = gr.File(label="Descargar JSON", interactive=False)
                        csv_export = gr.File(label="Descargar CSV", interactive=False)
            batch_status = gr.Markdown(elem_classes=["ti-run-status"])
            with gr.Group(elem_classes=["ti-pane", "ti-table-wrap"]):
                gr.HTML(pane_heading("Resultados del lote", "Predicción original y revisión humana por reseña"))
                batch_table = gr.Dataframe(headers=["ID", "Texto", "Modelo", "Predicción modelo", "Corrección humana", "Estado"],
                                           datatype=["str"] * 6, interactive=False, wrap=True, max_height=480,
                                           column_widths=[100, 440, 90, 145, 150, 126],
                                           label="Resultados reales del lote", elem_id="ti-batch-table")
                trend_table = gr.Dataframe(headers=["Predicción del modelo", "Cantidad"], datatype=["str", "number"], interactive=False, label="Conteos descriptivos de predicciones")
            batch_qwen = gr.HTML(label="Informes Qwen del lote · propuestas sin ejecutar")
        with gr.Tab("Agentes", id="agents", elem_id="ti-agents-tab"):
            gr.HTML(section_heading("Transparencia del sistema", "Coordinación multiagente", "Un orquestador y cuatro especialistas. Qwen/Ollama es un servicio opcional subordinado; sus estados se obtienen de la ejecución."))
            gr.HTML(AGENT_ORDER_HTML)
            execution_source = gr.Markdown("Sin ejecución registrada en esta sesión.")
            agents_panel = gr.HTML(render_agents_overview(), elem_id="ti-agents", label="Estados derivados de la traza real")
            with gr.Accordion("JSON completo de la ejecución mostrada", open=False, elem_classes=["ti-technical"]):
                agents_json = gr.Code(value="{}", label="Registro y procedencia originales", language="json", max_lines=18, wrap_lines=True, interactive=False, elem_id="ti-agents-json")
    gr.HTML('<div class="ti-footer">Traveler Insights 2026 · Universidad Tecnológica de Bolívar · Cartagena de Indias<br>Prototipo académico local · clasificación, evidencia y supervisión humana · sin acciones externas</div>')

    captured_outputs = [last_record, agents_panel, agents_json, execution_source]
    cleared_outputs = [inp, char_count, result, detail, agent_view, *captured_outputs]
    api_run.click(analyze_with_agents, inputs=[inp, model_choice, qwen_choice, manual], outputs=[result, detail, agent_view])
    captured = [result, detail, agent_view, *captured_outputs]
    editors = [inp, model_choice, qwen_choice, manual, translation, audio_input,
               example_mixed, example_positive, example_uncertain, run, clear, discard_audio, confirmed]
    def set_editable(enabled):
        return [gr.update(interactive=enabled) for _ in editors]

    def submit(review, kind, report, origin, language, edited, prior, reviewed, progress=gr.Progress()):
        yield (*set_editable(False), *clear_analysis()[2:], gr.skip(), gr.update(visible=True))
        if prior and prior.get("traduccion_automatica") and prior.get("advertencias") and edited and not reviewed:
            card, raw, trace = '<p role="alert">Confirma la revisión en Idioma y traducción experimental.</p>', "{}", prior
            trace = {**trace, "traduccion_editada": edited}
        else:
            card, raw, trace = unified(review, kind, report, origin, language, edited, prior,
                                       lambda stage: progress(None, desc=stage))
        record = json.loads(raw)
        updates = set_editable(True)
        updates[4] = gr.update(interactive=True, value=trace.get("traduccion_editada") or trace.get("traduccion_automatica") or "")
        updates[-1] = gr.update(interactive=True, value=False)
        technical = raw if record or not trace else json.dumps({"procesamiento_linguistico": trace}, ensure_ascii=False, indent=2)
        yield (*updates, card, technical, render_multiagent_view(record),
               *capture_execution(raw, "Audio · transcripción revisada" if origin else "Análisis individual"), trace, gr.update(visible=False))

    submission = run.click(submit, [inp, model_choice, qwen_choice, audio_metadata, manual, translation, previous, confirmed],
              [*editors, *captured, previous, cancel], trigger_mode="multiple", api_name=False)
    submission.then(fn=None, js="() => { delete document.documentElement.dataset.tiPending; }", api_name=False)
    def cancel_submission():
        return (*set_editable(True), *clear_analysis()[2:], gr.update(visible=False))
    cancel.click(cancel_submission, outputs=[*editors, *captured, cancel], cancels=[submission],
                 queue=False, js="() => { delete document.documentElement.dataset.tiPending; return []; }", api_name=False)

    def reset_input(name=None):
        values = load_example(name) if name else clear_analysis()
        return (*values, None, {}, "", "", {}, "", False)
    reset_outputs = [*cleared_outputs, audio_input, audio_metadata, manual, translation, previous, audio_status, confirmed]
    clear.click(reset_input, outputs=reset_outputs, api_name=False)
    for button, name in ((example_mixed, "Mixta"), (example_positive, "Positiva"), (example_uncertain, "Ambigua")):
        button.click(lambda name=name: reset_input(name), outputs=reset_outputs, api_name=False)
    inp.change(character_count, inp, char_count, queue=False, api_name=False)
    def invalidate_text(text, origin):
        if origin and not shares_transcript(text, origin["transcripcion_original"]):
            origin = {}
        return (*clear_analysis()[2:], origin, "", {}, "", False)
    inp.change(invalidate_text, [inp, audio_metadata], [*captured, audio_metadata, translation, previous, manual, confirmed], api_name=False)
    manual.input(lambda: ("", {}, False), outputs=[translation, previous, confirmed], api_name=False)
    gr.on([manual.input, translation.input, model_choice.input, qwen_choice.input],
          lambda: clear_analysis()[2:], outputs=captured, api_name=False)
    def start_transcription(path, language, text, progress=gr.Progress()):
        if text.strip():
            return None, "Ya hay texto. Pulsa Limpiar antes de grabar o cargar otra reseña."
        progress(None, desc="Transcribiendo")
        transcript, metadata, status = transcribe_audio_ui(path, language)
        return (transcript, metadata) if metadata else None, status

    def insert_transcription(pending, current):
        if not pending or current.strip():
            return (gr.skip(), gr.skip(), *[gr.skip()] * len(captured))
        return (*pending, *clear_analysis()[2:])

    transcription_event = gr.on([audio_input.stop_recording, audio_input.upload], start_transcription,
                                [audio_input, audio_language, inp], [pending_audio, audio_status], api_name=False)
    insertion = transcription_event.then(insert_transcription, [pending_audio, inp],
                                        [inp, audio_metadata, *captured], api_name=False)
    def discard(text, origin):
        values = list(reset_input())
        if not origin:
            values[:2] = [text, character_count(text)]
        return (*values, None)
    discard_audio.click(discard, [inp, audio_metadata], [*reset_outputs, pending_audio],
                        cancels=[transcription_event, insertion], api_name=False)
    clear.click(lambda: None, outputs=pending_audio, cancels=[transcription_event, insertion], api_name=False)
    audio_input.start_recording(lambda: "Grabando", outputs=audio_status, queue=False, api_name=False)
    theme_choice.change(fn=None, inputs=theme_choice, outputs=[], js=THEME_JS, queue=False, api_name=False)
    app.load(fn=lambda value: value, inputs=theme_choice, outputs=theme_choice, js=INITIAL_THEME_JS, queue=False, api_name=False)
    gr.on([model_choice.change, qwen_choice.change], configuration_context, [model_choice, qwen_choice], batch_config, api_name=False)
    batch_settings.click(lambda: gr.Tabs(selected="analysis"), outputs=navigation, queue=False, api_name=False)
    goto_agents.click(capture_execution, inputs=detail, outputs=captured_outputs, api_name=False).then(
        lambda: gr.Tabs(selected="agents"), outputs=navigation, api_name=False)
    batch_outputs = [batch_state, batch_table, trend_table, review_id, batch_status]
    run_lines.click(run_batch_lines, inputs=[batch_text, model_choice, qwen_choice], outputs=batch_outputs)
    run_csv.click(run_batch_csv, inputs=[batch_file, model_choice, qwen_choice], outputs=batch_outputs)
    batch_state.change(batch_generation_view, inputs=batch_state, outputs=batch_qwen)
    review_id.change(capture_batch_execution, inputs=[batch_state, review_id], outputs=captured_outputs, api_name=False)
    save_review.click(revise, inputs=[batch_state, review_id, corrected, note], outputs=batch_outputs).then(
        capture_batch_execution, inputs=[batch_state, review_id], outputs=captured_outputs, api_name=False)
    batch_goto_agents.click(capture_batch_execution, inputs=[batch_state, review_id], outputs=captured_outputs, api_name=False).then(
        lambda: gr.Tabs(selected="agents"), outputs=navigation, api_name=False)
    make_export.click(export_files, inputs=batch_state, outputs=[json_export, csv_export, batch_status])

if __name__ == "__main__":
    _LAUNCH_THEME = {"theme": THEME, "css": UI_CSS} if _GRADIO_MAJOR >= 6 else {}
    app.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", share=False, **_LAUNCH_THEME)
