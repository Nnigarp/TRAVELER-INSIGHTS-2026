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
from traveler_agent.multiagent_view import render_multiagent_view
from traveler_agent.audio_asr import AudioInputError, recognize_audio, analyze_edited_transcript
from traveler_agent.batch import (
    analyze_batch, apply_human_correction, demo_trends, export_csv,
    export_json, parse_review_csv, parse_review_lines,
)
from traveler_agent.demo_data import DEMO
from traveler_agent.model import MODEL_KINDS, load_model, save_model, train
from traveler_agent.model_runtime import configured_artifact, run_selected

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



# Midnight Indigo theme: use explicit light/dark colors and Gradio native tokens.
_INDIGO = gr.themes.colors.indigo
_PRIMARY = gr.themes.Color(
    c50=_INDIGO.c50, c100=_INDIGO.c100, c200=_INDIGO.c200,
    c300="#818CF8", c400=_INDIGO.c400, c500="#4F46E5",
    c600=_INDIGO.c600, c700=_INDIGO.c700, c800=_INDIGO.c800,
    c900=_INDIGO.c900, c950=_INDIGO.c950, name="midnight-indigo",
)
THEME = gr.themes.Soft(primary_hue=_PRIMARY, secondary_hue=_PRIMARY,
                       neutral_hue=gr.themes.colors.slate,
                       font=["ui-sans-serif", "system-ui", "sans-serif"]).set(
    body_background_fill="#F8FAFC",
    body_background_fill_dark="#0F172A",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#1E293B",
    body_text_color="#1E293B",
    body_text_color_dark="#F1F5F9",
    background_fill_primary="#FFFFFF",
    background_fill_primary_dark="#1E293B",
    background_fill_secondary="#F1F5F9",
    background_fill_secondary_dark="#172033",
)
# Structural CSS only; contrast and accent follow the current theme.
UI_CSS = """
.ti-hero {padding:1.25rem; border:1px solid var(--border-color-primary);
          border-radius:1rem; background:var(--background-fill-secondary);}
.ti-flow { color:var(--body-text-color); display:grid; gap:1rem; margin:.5rem 0 1rem; }
.ti-flow-lead,.ti-flow-agent,.ti-flow-outcome,.ti-flow-empty {
 border:1px solid var(--border-color-primary); border-radius:12px;
 background:var(--block-background-fill); padding:1rem; }
.ti-flow-lead {background:var(--background-fill-secondary);}
.ti-flow-lead h3 {margin:.25rem 0;}
.ti-flow-eyebrow {font-size:.75rem;letter-spacing:.075em;font-weight:650;}
.ti-flow-status {border:1px solid var(--border-color-primary);
 border-radius:999px;padding:.15rem .55rem;font-size:.78rem;
 background:var(--background-fill-secondary);font-weight:600;
 display:inline-block;white-space:normal;}
.ti-flow-agents {display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:.7rem;}
.ti-flow-agent p {margin:.4rem 0;color:var(--body-text-color-subdued);
 font-size:.875rem;}
.ti-flow-events {list-style:none;padding:0;margin:0;display:grid;gap:.45rem;}
.ti-flow-event {display:flex;align-items:center;gap:.7rem;padding:.65rem .8rem;
 border:1px solid var(--border-color-primary);border-radius:10px;
 background:var(--block-background-fill);flex-wrap:wrap;}
.ti-flow-event>div {flex:1;min-width:165px;}
.ti-flow-event-note,.ti-flow-event-task {font-size:.82rem;opacity:.85;}
.ti-flow-sequence {display:inline-grid;place-items:center;min-width:2rem;
 min-height:2rem;border-radius:9px;background:var(--background-fill-secondary);
 font-variant-numeric:tabular-nums;font-weight:600;}
.ti-flow-caveat {font-size:.83rem;opacity:.85;}
@media (max-width:600px) {.ti-flow-agents {grid-template-columns:1fr 1fr;}
 .ti-flow-event {align-items:flex-start;}}
@media (max-width:380px) {.ti-flow-agents {grid-template-columns:1fr;}}
"""

def _analysis_card(record: dict, *, model_kind: str, audio: bool = False) -> str:
    """Human readable summary only; source JSON and inference remain unchanged."""
    scores = record["puntuaciones_modelo_no_calibradas"]
    values = " · ".join(
        f"**{label}:** {float(scores[label]):.4f}"
        for label in ("negativo", "neutral", "positivo") if label in scores
    )
    aspects = ", ".join(record["aspectos_exploratorios"]) or "No identificados por reglas"
    review = "Sí, requiere revisión" if record["revision_humana"] else "No activada"
    sections = [
        "### Resumen de la experiencia",
        f"**Sentimiento {'experimental' if model_kind == 'beto' else 'DEMO'}:** {record['sentimiento']} · **Modelo:** {model_kind.upper()}",
        f"**Puntuaciones no calibradas:** {values}",
        f"**Aspectos exploratorios:** {aspects}",
        f"**Revisión humana:** {review}",
        f"**Recomendación exploratoria:** {record['recomendacion']}",
    ]
    if audio:
        is_corrected = record["procedencia_entrada"]["fue_corregida"]
        sections.insert(2, "**Transcripción corregida:** " + ("Sí" if is_corrected else "No"))
    sections.append("*BETO entrenado con datos sintéticos experimentales; consultar su procedencia.*"
                    if model_kind == 'beto' else
                    "*Modelo DEMO entrenado con ejemplos ficticios. "
                    "Las puntuaciones no acreditan precisión turística.*")
    selection = record.get("seleccion_modelo")
    if selection:
        if selection["fallback"]:
            sections.append("**BETO no completó el análisis; resultado ejecutado con LR de respaldo.**")
        else:
            sections.append("**BETO supervisado experimental, entrenado con datos sintéticos. "
                            "La regla histórica de revisión no valida automatizar sus decisiones.**")
            sections.append("*BETO usa una ventana limitada de tokens: una reseña larga "
                            "puede recortarse aunque cumpla el límite de caracteres. "
                            "Comprueba el contexto completo durante la revisión humana.*")
    return "\n\n".join(sections)

def analyze(review: str, model_kind: str = "lr"):
    try:
        record, effective = run_selected(model_kind, get_demo_model,
                                        lambda classifier: inspect_review(review, classifier))
        record["modelo_utilizado"] = effective
        summary = _analysis_card(record, model_kind=effective)
        return summary, json.dumps(record, ensure_ascii=False, indent=2)
    except (ValueError, TypeError) as exc:
        return f"Error de validación: {exc}", "{}"


def analyze_with_agents(review: str, model_kind: str = "lr"):
    """UI-only adapter: exactly one inference, unchanged legacy JSON outputs.

    A safe failure currently has no returned execution trace. Do not fabricate
    specialist status or a completed run when the underlying agent raised.
    """
    summary, json_text = analyze(review, model_kind)
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


def run_batch_lines(raw: str, model_kind: str = "lr"):
    try:
        reviews = parse_review_lines(raw)
        records, effective = run_selected(model_kind, get_demo_model,
            lambda classifier: analyze_batch(reviews, classifier, model_kind=model_kind))
        status = "Lote de demostración analizado."
        if model_kind == "beto": status += " BETO experimental." if effective == "beto" else " BETO falló; se ejecutó LR de respaldo."
        return _render_batch(records) + (status,)
    except ValueError as exc:
        return _render_batch([]) + (f"Error de validación: {exc}",)


def run_batch_csv(filepath: str | None, model_kind: str = "lr"):
    if not filepath:
        return _render_batch([]) + ("Selecciona un CSV UTF-8 con columna 'reseña'.",)
    try:
        path = Path(filepath)
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("El archivo supera el límite de 1 MiB.")
        reviews = parse_review_csv(path.read_bytes())
        records, effective = run_selected(model_kind, get_demo_model,
            lambda classifier: analyze_batch(reviews, classifier, model_kind=model_kind))
        status = "CSV de demostración analizado."
        if model_kind == "beto": status += " BETO experimental." if effective == "beto" else " BETO falló; se ejecutó LR de respaldo."
        return _render_batch(records) + (status,)
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
    csv_path.write_text(export_csv(records), encoding="utf-8-sig")
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


def analyze_audio_ui(edited_text: str, provenance: dict, model_kind: str):
    try:
        result, effective = run_selected(model_kind, get_demo_model,
            lambda classifier: analyze_edited_transcript(edited_text, provenance, classifier,
                                                        model_kind=model_kind))
        summary = _analysis_card(result, model_kind=effective, audio=True)
        return summary, json.dumps(result, ensure_ascii=False, indent=2)
    except (AudioInputError, ValueError, TypeError) as exc:
        return f"Error: {exc}", "{}"


def reset_audio_trace():
    """Do not analyze stale transcripts after replacing a recording."""
    return "", {}, "", "", "{}"

# Gradio 6 accepts launch-time theme/css; 4–5 attach them to Blocks.
_GRADIO_MAJOR = int(gr.__version__.split(".")[0])
_BLOCKS_THEME = {"theme": THEME, "css": UI_CSS} if _GRADIO_MAJOR < 6 else {}
with gr.Blocks(title="Traveler Insights 2026 | Demo académico", **_BLOCKS_THEME) as app:
    with gr.Group(elem_classes=["ti-hero"]):
        gr.Markdown(
            "**TRAVELER INSIGHTS · 2026**\n"
            "# Experiencias turísticas, análisis responsable\n"
            "Explora reseñas y voz mediante modelos locales, evidencias descriptivas "
            "y decisiones que priorizan la revisión humana.\n\n"
            "**PROTOTIPO ACADÉMICO · MODELOS DEMO:** entrenados con ejemplos ficticios; "
            "sin métricas científicas ni probabilidades externamente calibradas."
        )
    _MODEL_CHOICES = [("TF-IDF + regresión logística", "lr"), ("TF-IDF + SVM lineal", "svm")]
    if configured_artifact(): _MODEL_CHOICES.append(("BETO supervisado (experimental sintético)", "beto"))
    model_choice = gr.Dropdown(label="Modelo de demostración", choices=_MODEL_CHOICES, value="lr")
    gr.Markdown("Los modelos clásicos se entrenan con ejemplos **ficticios**. La selección no demuestra rendimiento ni calibración externa.")
    if configured_artifact():
        gr.Markdown("BETO es un clasificador supervisado **experimental sintético**. "
                    "Su selección es opcional; LR permanece como valor inicial y respaldo explícito.")
    with gr.Tab("Reseña individual"):
        gr.Markdown("### Interpretar una experiencia\nEscribe una reseña para ver inferencia y evidencia descriptiva.")
        inp = gr.Textbox(label="Reseña turística", lines=4,
                         value="El guía fue amable, pero esperamos demasiado para entrar.")
        run = gr.Button("Analizar reseña", variant="primary")
        result = gr.Markdown(label="Resumen visual del análisis")
        gr.Markdown("### Coordinación de los cinco agentes")
        agent_view = gr.HTML(label="Orquestador y delegaciones verificadas")
        with gr.Accordion("Trazabilidad completa · JSON", open=False):
            detail = gr.Code(label="JSON sin modificaciones", language="json")
        run.click(
            analyze_with_agents, inputs=[inp, model_choice],
            outputs=[result, detail, agent_view],
        )


    with gr.Tab("Audio y transcripción local"):
        gr.Markdown("### Voz → transcripción → corrección → análisis\nGraba o sube un archivo compatible y revisa el texto antes de clasificar.")
        gr.Markdown(
            "**Solo reconocimiento de voz:** no detecta emociones por la voz. "
            "Micrófono o archivo convertido por Gradio a WAV PCM; máximo 120 s y 12 MiB "
            "después de la conversión. La primera descarga de pesos requiere autorización "
            "explícita y el complemento ASR opcional. El texto se puede corregir antes "
            "de usar el modelo seleccionado arriba: LR, SVM o BETO experimental."
        )
        audio_input = gr.Audio(
            label="Grabar voz o cargar archivo de audio",
            sources=["microphone", "upload"], type="filepath", format="wav",
        )
        audio_language = gr.Dropdown(
            label="Idioma de la voz", choices=[("Español", "es"), ("English", "en"),
                                              ("Italiano", "it"), ("Detectar", "auto")],
            value="es",
        )
        audio_transcribe_btn = gr.Button("Transcribir localmente")
        audio_status = gr.Markdown()
        audio_transcript = gr.Textbox(
            label="Transcripción editable — revisa negaciones y palabras mal reconocidas",
            lines=5, interactive=True,
        )
        audio_metadata = gr.State({})
        audio_analyze_btn = gr.Button("Analizar transcripción corregida", variant="primary")
        audio_summary = gr.Markdown(label="Resumen de la transcripción analizada")
        with gr.Accordion("Procedencia ASR y análisis · JSON", open=False):
            audio_trace = gr.Code(label="ASR original + edición + análisis", language="json")
        audio_transcribe_btn.click(
            transcribe_audio_ui, inputs=[audio_input, audio_language],
            outputs=[audio_transcript, audio_metadata, audio_status],
        )
        audio_analyze_btn.click(
            analyze_audio_ui, inputs=[audio_transcript, audio_metadata, model_choice],
            outputs=[audio_summary, audio_trace],
        )
        audio_input.change(
            reset_audio_trace, outputs=[
                audio_transcript, audio_metadata, audio_status, audio_summary, audio_trace
            ]
        )
        audio_language.change(
            reset_audio_trace, outputs=[
                audio_transcript, audio_metadata, audio_status, audio_summary, audio_trace
            ]
        )

    with gr.Tab("Lotes y revisión humana"):
        gr.Markdown("### Analizar varias reseñas\nObserva tendencias descriptivas y conserva las etiquetas originales antes de exportar.")
        gr.Markdown("Una reseña por línea (máximo 200) **o** CSV UTF-8 con columna `reseña` (máximo 1 MiB). "
                    "Las tendencias son conteos descriptivos de predicciones DEMO, **no** métricas de exactitud.")
        batch_state = gr.State([])
        batch_text = gr.Textbox(label="Reseñas de demostración (una por línea)", lines=6,
                                value="La visita fue increíble 😍\nEl precio era excesivo 😡\nEl tour duró dos horas.")
        run_lines = gr.Button("Analizar texto por lotes", variant="primary")
        batch_file = gr.File(label="O cargar CSV local", file_types=[".csv"], type="filepath")
        run_csv = gr.Button("Analizar CSV")
        batch_status = gr.Markdown()
        gr.Markdown("#### Resultados por reseña")
        batch_table = gr.Dataframe(headers=["ID", "Texto", "Modelo", "Predicción modelo", "Corrección humana", "Estado"],
                                   datatype=["str"] * 6, interactive=False, label="Resultados de demostración")
        gr.Markdown("#### Tendencias de predicciones DEMO")
        trend_table = gr.Dataframe(headers=["Predicción del modelo", "Cantidad"],
                                   datatype=["str", "number"], interactive=False,
                                   label="Tendencias de demostración")
        gr.Markdown("### Corrección humana\nLa clasificación original se conserva; "
                    "la corrección no reentrena el modelo ni genera acciones automáticas.")
        review_id = gr.Dropdown(label="ID de reseña", choices=[])
        corrected = gr.Dropdown(label="Etiqueta revisada", choices=["positivo", "neutral", "negativo"], value="neutral")
        note = gr.Textbox(label="Motivo opcional (máximo 500 caracteres)")
        save_review = gr.Button("Registrar corrección")
        make_export = gr.Button("Preparar exportaciones")
        json_export = gr.File(label="Descargar JSON de demostración", interactive=False)
        csv_export = gr.File(label="Descargar CSV de demostración", interactive=False)
        outputs = [batch_state, batch_table, trend_table, review_id, batch_status]
        run_lines.click(run_batch_lines, inputs=[batch_text, model_choice], outputs=outputs)
        run_csv.click(run_batch_csv, inputs=[batch_file, model_choice], outputs=outputs)
        save_review.click(revise, inputs=[batch_state, review_id, corrected, note], outputs=outputs)
        make_export.click(export_files, inputs=batch_state, outputs=[json_export, csv_export, batch_status])

if __name__ == "__main__":
    _LAUNCH_THEME = {"theme": THEME, "css": UI_CSS} if _GRADIO_MAJOR >= 6 else {}
    app.launch(server_name="127.0.0.1", share=False, **_LAUNCH_THEME)
