"""Local demo UI: individual/batch analysis with auditable human corrections.

No external sharing, scientific metrics, or automatic online learning.
"""
from __future__ import annotations

import json
from functools import lru_cache
import tempfile
from pathlib import Path
import gradio as gr
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



from traveler_agent.ui_style import THEME, UI_CSS, THEME_JS, INITIAL_THEME_JS, _PRIMARY
from traveler_agent.ui_presenter import (
    BRAND_HTML, HERO_HTML, EMPTY_REPORT_HTML, render_result, section_heading, pane_heading,
)


def _analysis_card(record: dict, *, model_kind: str, audio: bool = False) -> str:
    return render_result(record, model_kind=model_kind, audio=audio)


def analyze(review: str, model_kind: str = "lr", qwen_enabled: bool = False):
    try:
        record, effective = run_selected(model_kind, get_demo_model,
                                        lambda classifier: inspect_review(review, classifier, qwen_enabled=qwen_enabled))
        record["modelo_utilizado"] = effective
        summary = _analysis_card(record, model_kind=effective)
        return summary, json.dumps(record, ensure_ascii=False, indent=2)
    except (ValueError, TypeError) as exc:
        return f"Error de validación: {exc}", "{}"


def analyze_with_agents(review: str, model_kind: str = "lr", qwen_enabled: bool = False):
    """UI-only adapter: exactly one inference, unchanged legacy JSON outputs.

    A safe failure currently has no returned execution trace. Do not fabricate
    specialist status or a completed run when the underlying agent raised.
    """
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
        text, provenance = recognize_audio(filepath, language)
        return text, provenance, (
            f"Transcripción local terminada ({provenance['duracion_segundos']} s; "
            f"idioma ASR: {provenance['idioma_detectado_asr']}). "
            "Revisa especialmente negaciones, nombres y ruido antes de analizar."
        )
    except AudioInputError as exc:
        return "", {}, f"Error de audio: {exc}"


def analyze_audio_ui(edited_text: str, provenance: dict, model_kind: str, qwen_enabled: bool = False):
    try:
        result, effective = run_selected(model_kind, get_demo_model,
            lambda classifier: analyze_edited_transcript(edited_text, provenance, classifier,
                                                        model_kind=model_kind, qwen_enabled=qwen_enabled))
        summary = _analysis_card(result, model_kind=effective, audio=True)
        return summary, json.dumps(result, ensure_ascii=False, indent=2)
    except (AudioInputError, ValueError, TypeError) as exc:
        return f"Error: {exc}", "{}"


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


def clear_analysis():
    return ("", character_count(""), EMPTY_REPORT_HTML, "{}", render_multiagent_view(None),
            *capture_execution("{}"))


def load_example(name):
    values = list(clear_analysis())
    values[0] = DEMO_EXAMPLES[name]
    values[1] = character_count(values[0])
    return tuple(values)


# Gradio 6 accepts launch-time theme/css; 4–5 attach them to Blocks.
_GRADIO_MAJOR = int(gr.__version__.split(".")[0])
_BLOCKS_THEME = {"theme": THEME, "css": UI_CSS} if _GRADIO_MAJOR < 6 else {}
_BETO_AVAILABLE = artifact_available(configured_artifact())
with gr.Blocks(title="Traveler Insights 2026 | Cartagena · UTB", analytics_enabled=False, **_BLOCKS_THEME) as app:
    last_record = gr.State(None)
    with gr.Row(elem_id="ti-topbar"):
        with gr.Column(scale=3, min_width=260):
            gr.HTML(BRAND_HTML)
        with gr.Column(scale=1, min_width=265):
            theme_choice = gr.Radio(["Caribe Nocturno", "UTB Claro"], value="Caribe Nocturno",
                                    label="Tema visual", elem_id="ti-theme")
    with gr.Tabs(selected="analysis", elem_id="ti-nav") as navigation:
        with gr.Tab("Análisis", id="analysis"):
            gr.HTML(HERO_HTML)
            with gr.Row(elem_id="ti-analysis-row"):
                with gr.Column(scale=4, min_width=300, elem_id="ti-input-pane", elem_classes=["ti-pane"]):
                    gr.HTML(pane_heading("Nueva evaluación", "Configura los modelos y analiza una reseña turística."))
                    _MODEL_CHOICES = [("TF-IDF · Regresión logística", "lr"), ("TF-IDF · SVM lineal", "svm")]
                    if _BETO_AVAILABLE:
                        _MODEL_CHOICES.append(("BETO V3 · Transformer experimental", "beto"))
                    model_choice = gr.Dropdown(label="Modelo de demostración", choices=_MODEL_CHOICES, value="lr", elem_id="ti-model")
                    qwen_choice = gr.Checkbox(label="Activar informe local Qwen3 1.7B (Ollama)", value=False,
                                             info="Generación opcional · evidencia y revisión protegidas", elem_id="ti-qwen")
                    inp = gr.Textbox(label="Reseña turística", lines=6, value="",
                                     placeholder="Describe la experiencia que deseas analizar…", elem_id="ti-review")
                    char_count = gr.Markdown(character_count(""), elem_id="ti-char-count")
                    gr.HTML('<p class="ti-example-note">Ejemplos demostrativos · solo cargan texto; no predicen el resultado.</p>')
                    with gr.Row(elem_classes=["ti-example-row"]):
                        example_mixed = gr.Button("Mixta · transporte", size="sm")
                        example_positive = gr.Button("Positiva · servicio", size="sm")
                        example_uncertain = gr.Button("Ambigua · regreso", size="sm")
                    with gr.Row():
                        run = gr.Button("Analizar reseña", variant="primary", elem_id="ti-analyze")
                        clear = gr.Button("Limpiar", elem_id="ti-clear")
                    gr.HTML('<div class="ti-input-note"><strong>Inferencia local real.</strong> LR/SVM usan datos ficticios y BETO es experimental. Qwen no cambia clasificación ni revisión humana. No se ejecutan acciones externas.</div>')
                with gr.Column(scale=6, min_width=300, elem_id="ti-result-pane", elem_classes=["ti-pane"]):
                    gr.HTML(pane_heading("Informe de experiencia", "Síntesis, evidencia y propuestas para valoración humana."))
                    result = gr.HTML(EMPTY_REPORT_HTML, elem_id="ti-result", label="Informe de experiencia")
                    with gr.Accordion("Detalles técnicos · JSON y trazabilidad", open=False):
                        detail = gr.Code(value="{}", label="JSON completo sin modificaciones", language="json", elem_id="ti-detail-json")
                        agent_view = gr.HTML(render_multiagent_view(None), label="Traza de esta ejecución")
                    goto_agents = gr.Button("Explorar trazabilidad", size="sm")
        with gr.Tab("Audio", id="audio"):
            gr.HTML(section_heading("Entrada multimodal", "Audio y transcripción", "Graba o carga audio, revisa las palabras y analiza con el modelo y Qwen seleccionados en Análisis."))
            with gr.Row():
                with gr.Column(min_width=300, elem_classes=["ti-pane"]):
                    gr.HTML(pane_heading("Tu voz, como entrada", "Whisper Tiny local · micrófono o archivo · transcripción editable"))
                    audio_input = gr.Audio(label="Grabar voz o cargar archivo de audio", sources=["microphone", "upload"], type="filepath", format="wav", elem_id="ti-audio-input")
                    audio_language = gr.Dropdown(label="Idioma de la voz", choices=[("Español", "es"), ("English", "en"), ("Italiano", "it"), ("Detectar", "auto")], value="es")
                    audio_transcribe_btn = gr.Button("Transcribir localmente", variant="primary")
                    audio_status = gr.Markdown()
                    audio_transcript = gr.Textbox(label="Transcripción editable — revisa negaciones y palabras mal reconocidas", lines=5, interactive=True)
                    audio_metadata = gr.State({})
                    audio_analyze_btn = gr.Button("Analizar transcripción corregida", variant="primary")
                    gr.HTML('<div class="ti-input-note">Solo reconocimiento de palabras; no se infiere emoción desde la voz. WAV PCM de 0,25–120 s y hasta 12 MiB tras conversión. Conserva tu caché Tiny existente; la primera descarga necesita autorización explícita.</div>')
                with gr.Column(min_width=300, elem_classes=["ti-pane"]):
                    gr.HTML(pane_heading("Informe de la transcripción", "Procedencia ASR y edición humana conservadas."))
                    audio_summary = gr.HTML(EMPTY_REPORT_HTML, label="Informe de audio")
                    with gr.Accordion("Procedencia ASR y análisis · JSON", open=False):
                        audio_trace = gr.Code(value="{}", label="ASR original + edición + análisis", language="json", elem_id="ti-audio-json")
                    audio_goto_agents = gr.Button("Ver agentes de la última ejecución", size="sm")
        with gr.Tab("Lotes", id="batch"):
            gr.HTML(section_heading("Revisión de varias experiencias", "Análisis por lotes", "Procesa texto o CSV, conserva predicciones y correcciones y exporta resultados reales. Los conteos son descriptivos."))
            batch_state = gr.State([])
            with gr.Row():
                with gr.Column(min_width=300, elem_classes=["ti-pane"]):
                    gr.HTML(pane_heading("Reseñas del lote", "Máximo 200 reseñas · CSV UTF-8 de hasta 1 MiB con columna reseña"))
                    batch_text = gr.Textbox(label="Reseñas (una por línea)", lines=7, placeholder="Una reseña por línea. No hay resultados hasta analizar.", elem_id="ti-batch-text")
                    run_lines = gr.Button("Analizar texto por lotes", variant="primary")
                    batch_file = gr.File(label="Cargar CSV local", file_types=[".csv"], type="filepath")
                    run_csv = gr.Button("Analizar CSV")
                    gr.Markdown("Qwen consulta hasta cinco reseñas en secuencia con presupuesto compartido. El resto conserva análisis y recomendación por reglas.")
                with gr.Column(min_width=300, elem_classes=["ti-pane"]):
                    gr.HTML(pane_heading("Revisión humana", "La corrección se registra sin cambiar ni reentrenar el modelo."))
                    review_id = gr.Dropdown(label="ID de reseña", choices=[])
                    corrected = gr.Dropdown(label="Etiqueta revisada", choices=["positivo", "neutral", "negativo"], value="neutral", elem_id="ti-correction")
                    note = gr.Textbox(label="Motivo opcional (máximo 500 caracteres)")
                    save_review = gr.Button("Registrar corrección", variant="primary")
                    batch_goto_agents = gr.Button("Ver traza de la reseña seleccionada")
                    make_export = gr.Button("Preparar exportaciones")
                    with gr.Row():
                        json_export = gr.File(label="Descargar JSON", interactive=False)
                        csv_export = gr.File(label="Descargar CSV", interactive=False)
            batch_status = gr.Markdown()
            with gr.Group(elem_classes=["ti-pane", "ti-table-wrap"]):
                gr.HTML(pane_heading("Resultados del lote", "Predicción original y revisión humana por reseña"))
                batch_table = gr.Dataframe(headers=["ID", "Texto", "Modelo", "Predicción modelo", "Corrección humana", "Estado"], datatype=["str"] * 6, interactive=False, label="Resultados reales del lote")
                trend_table = gr.Dataframe(headers=["Predicción del modelo", "Cantidad"], datatype=["str", "number"], interactive=False, label="Conteos descriptivos de predicciones")
            batch_qwen = gr.HTML(label="Informes Qwen del lote · propuestas sin ejecutar")
        with gr.Tab("Agentes", id="agents"):
            gr.HTML(section_heading("Transparencia del sistema", "Coordinación multiagente", "Un orquestador y cuatro especialistas. Qwen/Ollama es un servicio opcional subordinado; sus estados se obtienen de la ejecución."))
            execution_source = gr.Markdown("Sin ejecución registrada en esta sesión.")
            agents_panel = gr.HTML(render_agents_overview(), elem_id="ti-agents", label="Estados derivados de la traza real")
            with gr.Accordion("JSON completo de la ejecución mostrada", open=False):
                agents_json = gr.Code(value="{}", label="Registro y procedencia originales", language="json", elem_id="ti-agents-json")
    gr.HTML('<div class="ti-footer">Traveler Insights 2026 · Universidad Tecnológica de Bolívar · Cartagena de Indias<br>Prototipo académico local · clasificación, evidencia y supervisión humana · sin acciones externas</div>')

    captured_outputs = [last_record, agents_panel, agents_json, execution_source]
    cleared_outputs = [inp, char_count, result, detail, agent_view, *captured_outputs]
    run.click(analyze_with_agents, inputs=[inp, model_choice, qwen_choice], outputs=[result, detail, agent_view]).then(
        capture_execution, inputs=detail, outputs=captured_outputs, api_name=False)
    inp.input(character_count, inputs=inp, outputs=char_count, queue=False, api_name=False)
    clear.click(clear_analysis, outputs=cleared_outputs, api_name=False)
    example_mixed.click(lambda: load_example("Mixta"), outputs=cleared_outputs, api_name=False)
    example_positive.click(lambda: load_example("Positiva"), outputs=cleared_outputs, api_name=False)
    example_uncertain.click(lambda: load_example("Ambigua"), outputs=cleared_outputs, api_name=False)
    theme_choice.change(fn=None, inputs=theme_choice, outputs=[], js=THEME_JS, queue=False, api_name=False)
    app.load(fn=None, inputs=None, outputs=[], js=INITIAL_THEME_JS, api_name=False)
    for button in (goto_agents, audio_goto_agents):
        button.click(lambda: gr.Tabs(selected="agents"), outputs=navigation, api_name=False)
    audio_transcribe_btn.click(transcribe_audio_ui, inputs=[audio_input, audio_language], outputs=[audio_transcript, audio_metadata, audio_status])
    audio_analyze_btn.click(analyze_audio_ui, inputs=[audio_transcript, audio_metadata, model_choice, qwen_choice], outputs=[audio_summary, audio_trace]).then(
        capture_execution, inputs=audio_trace, outputs=captured_outputs, api_name=False)
    for component in (audio_input, audio_language):
        component.change(reset_audio_trace, outputs=[audio_transcript, audio_metadata, audio_status, audio_summary, audio_trace])
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
