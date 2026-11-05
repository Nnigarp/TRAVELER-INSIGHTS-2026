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
from traveler_agent.audio_asr import AudioInputError, recognize_audio, analyze_edited_transcript
from traveler_agent.batch import (
    analyze_batch, apply_human_correction, demo_trends, export_csv,
    export_json, parse_review_csv, parse_review_lines,
)
from traveler_agent.demo_data import DEMO
from traveler_agent.model import MODEL_KINDS, load_model, save_model, train

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


def analyze(review: str, model_kind: str = "lr"):
    try:
        record = inspect_review(review, get_demo_model(model_kind))
        record["modelo_utilizado"] = model_kind
        summary = (f"**Sentimiento:** {record['sentimiento']} | "
                   f"**Revisión humana:** {'Sí' if record['revision_humana'] else 'No'}\n\n"
                   f"**Sugerencia:** {record['recomendacion']}")
        return summary, json.dumps(record, ensure_ascii=False, indent=2)
    except (ValueError, TypeError) as exc:
        return f"Error de validación: {exc}", "{}"


def _render_batch(records: list[dict]):
    table = [[r["id"], r["texto_original"], r.get("modelo_utilizado", "lr"), r["sentimiento"],
              r["etiqueta_corregida_humano"] or "", r["estado_revision"]]
             for r in records]
    counts = [[x["sentimiento_modelo"], x["cantidad"]] for x in demo_trends(records)]
    ids = [r["id"] for r in records]
    return records, table, counts, gr.update(choices=ids, value=ids[0] if ids else None)


def run_batch_lines(raw: str, model_kind: str = "lr"):
    try:
        return _render_batch(analyze_batch(parse_review_lines(raw), get_demo_model(model_kind), model_kind=model_kind)) + ("Lote de demostración analizado.",)
    except ValueError as exc:
        return _render_batch([]) + (f"Error de validación: {exc}",)


def run_batch_csv(filepath: str | None, model_kind: str = "lr"):
    if not filepath:
        return _render_batch([]) + ("Selecciona un CSV UTF-8 con columna 'reseña'.",)
    try:
        path = Path(filepath)
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("El archivo supera el límite de 1 MiB.")
        return _render_batch(analyze_batch(parse_review_csv(path.read_bytes()), get_demo_model(model_kind), model_kind=model_kind)) + ("CSV de demostración analizado.",)
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
        result = analyze_edited_transcript(
            edited_text, provenance, get_demo_model(model_kind),
            model_kind=model_kind,
        )
        summary = (
            f"**Sentimiento DEMO:** {result['sentimiento']} | "
            f"**Modelo:** {model_kind} | "
            f"**Transcripción corregida:** "
            f"{'Sí' if result['procedencia_entrada']['fue_corregida'] else 'No'}"
            f"\n\n**Recomendación exploratoria:** {result['recomendacion']}"
        )
        return summary, json.dumps(result, ensure_ascii=False, indent=2)
    except (AudioInputError, ValueError, TypeError) as exc:
        return f"Error: {exc}", "{}"


def reset_audio_trace():
    """Do not analyze stale transcripts after replacing a recording."""
    return "", {}, "", "", "{}"

with gr.Blocks(title="Traveler Insights 2026 | Demo académico") as app:
    gr.Markdown(
        "# Traveler Insights 2026 · Prototipo académico\n"
        "**AVISO:** modelo entrenado con ejemplos ficticios; no corresponde a Kaggle ni acredita precisión científica. "
        "La aplicación es local y las puntuaciones no están calibradas."
    )
    model_choice = gr.Dropdown(label="Modelo de demostración", choices=[("TF-IDF + regresión logística", "lr"), ("TF-IDF + SVM lineal", "svm")], value="lr")
    gr.Markdown("Ambos modelos se entrenan con ejemplos **ficticios**. La selección no demuestra rendimiento ni calibración externa.")
    with gr.Tab("Reseña individual"):
        inp = gr.Textbox(label="Reseña turística", lines=4,
                         value="El guía fue amable, pero esperamos demasiado para entrar.")
        run = gr.Button("Analizar reseña", variant="primary")
        result = gr.Markdown(label="Resultado")
        detail = gr.Code(label="Trazabilidad JSON", language="json")
        run.click(analyze, inputs=[inp, model_choice], outputs=[result, detail])


    with gr.Tab("Audio y transcripción local"):
        gr.Markdown(
            "**Solo reconocimiento de voz:** no detecta emociones por la voz. "
            "Micrófono o archivo convertido por Gradio a WAV PCM; máximo 120 s y 12 MiB "
            "después de la conversión. La primera descarga de pesos requiere autorización "
            "explícita y el complemento ASR opcional. El texto se puede corregir antes "
            "de usar el mismo modelo LR/SVM seleccionado arriba."
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
        audio_summary = gr.Markdown()
        audio_trace = gr.Code(label="Trazabilidad: ASR original + corrección + análisis", language="json")
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
        gr.Markdown("Una reseña por línea (máximo 200) **o** CSV UTF-8 con columna `reseña` (máximo 1 MiB). "
                    "Las tendencias son conteos descriptivos de predicciones DEMO, **no** métricas de exactitud.")
        batch_state = gr.State([])
        batch_text = gr.Textbox(label="Reseñas de demostración (una por línea)", lines=6,
                                value="La visita fue increíble 😍\nEl precio era excesivo 😡\nEl tour duró dos horas.")
        run_lines = gr.Button("Analizar texto por lotes", variant="primary")
        batch_file = gr.File(label="O cargar CSV local", file_types=[".csv"], type="filepath")
        run_csv = gr.Button("Analizar CSV")
        batch_status = gr.Markdown()
        batch_table = gr.Dataframe(headers=["ID", "Texto", "Modelo", "Predicción modelo", "Corrección humana", "Estado"],
                                   datatype=["str"] * 6, interactive=False, label="Resultados de demostración")
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
    app.launch(server_name="127.0.0.1", share=False)
