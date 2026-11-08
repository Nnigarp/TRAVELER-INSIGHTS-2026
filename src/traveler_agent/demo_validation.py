"""Executed functional evidence for the DEMO, never a scientific benchmark.

Runs the existing classifier/agents and optional Gradio callbacks. Fixtures are
explicitly artificial; no accuracy estimate, threshold selection or human data
admission is performed. Importing this module has no model, audio or UI effects.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
from typing import Any

from .agent import inspect_review
from .batch import (analyze_batch, apply_human_correction, export_csv, export_json,
                    parse_review_csv, parse_review_lines)
from .demo_data import DEMO
from .model import train

# These are functional examples, not human gold, a new corpus, or held-out tests.
CASES = (
    ("positive_language", "La atención fue excelente, disfruté la visita y volvería encantado."),
    ("negative_language", "Nos dejaron esperando y la habitación estaba sucia; fue una experiencia pésima."),
    ("neutral_description", "La visita empezó a las nueve y terminó al mediodía."),
    ("mixed_aspects", "La comida estuvo deliciosa, pero el local era ruidoso y la atención tardó bastante."),
    # Deliberately retain the previously reported manual limitation as evidence.
    ("known_complex_negation", "No fue una mala experiencia. El servicio no estuvo nada mal y volvería a visitar el lugar."),
    ("review_route", "El guía fue amable, pero esperamos demasiado para entrar."),
)
SPECIALISTS = ("percepcion", "semantica", "control", "accion")


def trace_violations(record: dict) -> list[str]:
    """Verify real returned events, transitions, outputs and literal evidence."""
    problems = []
    meta = record.get("multiagente", {})
    trace = meta.get("traza", [])
    if meta.get("orquestador") != "TravelerOrchestrator":
        problems.append("orchestrator_missing")
    if meta.get("plan") != list(SPECIALISTS):
        problems.append("plan_incomplete")
    if [x.get("agent") for x in trace if x.get("status") == "OK"] != list(SPECIALISTS):
        problems.append("specialist_execution_incomplete")
    if len(trace) != 10 or [x.get("sequence") for x in trace] != list(range(1, 11)):
        problems.append("trace_sequence_invalid")
    handoffs = [x.get("task") for x in trace if x.get("status") == "DELEGATED"]
    if handoffs != list(SPECIALISTS):
        problems.append("handoff_incomplete")
    if record.get("sentimiento") not in {"negativo", "neutral", "positivo"}:
        problems.append("invalid_sentiment")
    review = record.get("revision_humana")
    if not isinstance(review, bool):
        problems.append("invalid_review_flag")
    expected_state = "REVISION_HUMANA" if review else "COMPLETADO"
    if meta.get("estado") != expected_state:
        problems.append("final_state_inconsistent")
    if review and record.get("tipo_salida") != "revisión_humana":
        problems.append("human_review_bypassed")
    if meta.get("acciones_externas") is not False:
        problems.append("external_action_reported")
    return problems


def _observed(case_id: str, text: str, model: Any) -> dict:
    start = time.perf_counter()
    record = inspect_review(text, model, review_id="functional-" + case_id)
    milliseconds = (time.perf_counter() - start) * 1000
    violations = trace_violations(record)
    if any(span not in text for span in record["fragmentos_literales_no_explicativos"]):
        violations.append("evidence_not_literal")
    if text in json.dumps(record["multiagente"]["traza"], ensure_ascii=False):
        violations.append("raw_text_in_metadata_trace")
    return {"case_id": case_id, "fixture_text": text,
            "fixture_origin": "synthetic_functional_example_not_human_gold",
            "latency_ms": round(milliseconds, 3), "contract_violations": violations,
            "observed": record}


def _resource_snapshot() -> dict:
    result = {"cpu_process_seconds": round(time.process_time(), 4)}
    try:
        import resource  # Unix high-water mark, not memory attributable to one model.
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["process_peak_rss_mib"] = round(value / (1024 * 1024 if platform.system() == "Darwin" else 1024), 3)
        result["memory_measurement"] = "OS_process_high_water_mark_includes_imports_and_all_models"
    except ImportError:
        try:
            import psutil
            info = psutil.Process().memory_info()
            if hasattr(info, "peak_wset"):
                result["process_peak_rss_mib"] = round(info.peak_wset / (1024 * 1024), 3)
                result["memory_measurement"] = "OS_process_peak_working_set_includes_all_models"
            else:
                result["process_current_rss_mib"] = round(info.rss / (1024 * 1024), 3)
                result["memory_measurement"] = "current_RSS_not_peak"
        except ImportError:
            result["memory_measurement"] = "PENDING_install_optional_psutil_for_Windows_measurement"
    return result


def _ui_callbacks(model_kind: str) -> dict:
    """Execute shipped Gradio callbacks without starting/publishing a server."""
    ui = importlib.import_module("app")
    violations = []
    records = []
    for case_id, text in CASES:
        _, encoded, rendered = ui.analyze_with_agents(text, model_kind)
        record = json.loads(encoded)
        violations.extend(case_id + ":" + code for code in trace_violations(record))
        if not rendered or "TravelerOrchestrator" not in rendered:
            violations.append(case_id + ":agent_view_missing")
        if model_kind == "beto" and record.get("modelo_utilizado") != "beto":
            violations.append(case_id + ":BETO_not_actually_run")
        records.append({"case_id": case_id, "sentiment": record.get("sentimiento"),
                        "human_review": record.get("revision_humana"),
                        "effective_model": record.get("modelo_utilizado"),
                        "model_selection": record.get("seleccion_modelo")})
    lines = "\n".join(text for _, text in CASES)
    batch, table, counts, _, _ = ui.run_batch_lines(lines, model_kind)
    if len(batch) != len(CASES) or len(table) != len(CASES) or sum(x[1] for x in counts) != len(CASES):
        violations.append("UI_batch_count_invalid")
    with tempfile.TemporaryDirectory(prefix="traveler_validation_csv_") as temporary:
        source = Path(temporary) / "functional.csv"
        data = io.StringIO(newline="")
        writer = csv.writer(data)
        writer.writerow(["reseña"])
        writer.writerows([[text] for _, text in CASES])
        source.write_text(data.getvalue(), encoding="utf-8-sig")
        imported, _, _, _, _ = ui.run_batch_csv(str(source), model_kind)
        if len(imported) != len(CASES):
            violations.append("UI_CSV_batch_invalid")
    if not batch:
        return {"status": "FAILED", "violations": violations + ["no_batch"], "observed": records}
    before = batch[0]["sentimiento"]
    corrected, *_ = ui.revise(batch, batch[0]["id"], "positivo", "Fixture funcional de corrección; no anotación humana real.")
    if corrected[0]["sentimiento"] != before or batch[0]["etiqueta_corregida_humano"] is not None:
        violations.append("UI_correction_changed_prediction_or_original")
    json_path, csv_path, _ = ui.export_files(corrected)
    # Only delete the temporary directory actually created by this callback.
    try:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        exported = list(csv.DictReader(io.StringIO(Path(csv_path).read_text(encoding="utf-8-sig"))))
        if payload.get("n") != len(CASES) or len(exported) != len(CASES):
            violations.append("UI_export_roundtrip_invalid")
        export_sizes = {"json_bytes": Path(json_path).stat().st_size,
                        "csv_bytes": Path(csv_path).stat().st_size}
    finally:
        directory = Path(json_path).parent
        if directory == Path(csv_path).parent and directory.name.startswith("traveler_demo_"):
            shutil.rmtree(directory)
    _, provenance, audio_status = ui.transcribe_audio_ui(None, "es")
    _, encoded = ui.analyze_audio_ui("Fixture sin audio previo", {}, model_kind)
    if provenance or "Error" not in audio_status or encoded != "{}":
        violations.append("UI_missing_audio_guard_invalid")
    return {"status": "PASSED_CALLBACKS_BROWSER_NOT_VALIDATED" if not violations else "FAILED",
            "violations": violations, "observed": records,
            "selector_options_at_startup": [value for _, value in ui._MODEL_CHOICES],
            "batch_rows": len(batch), "exports": export_sizes,
            "human_correction": "SIMULATED_FUNCTIONAL_ONLY_NOT_AN_ANNOTATION",
            "audio_guard": "EXECUTED_NO_AUDIO_INPUT",
            "browser_visual_audio_permissions": "PENDING_manual_Windows_browser_check"}


def _real_audio(filepath: Path, model: Any, model_kind: str) -> dict:
    # Private input text/transcript/path are intentionally omitted from this report.
    from .audio_asr import analyze_edited_transcript, recognize_audio
    start = time.perf_counter()
    transcript, provenance = recognize_audio(filepath, "es")
    asr_seconds = time.perf_counter() - start
    record = analyze_edited_transcript(transcript, provenance, model, model_kind=model_kind)
    return {"status": "EXECUTED_REAL_LOCAL_ASR_NOT_HUMAN_SENTIMENT_GOLD",
            "asr_seconds": round(asr_seconds, 4), "duration_seconds": provenance["duracion_segundos"],
            "backend": provenance["motor_asr"], "model": provenance["modelo_asr"],
            "transcript_characters": len(transcript), "sentiment": record["sentimiento"],
            "human_review": record["revision_humana"],
            "trace": record["multiagente"]["traza"],
            "violations": trace_violations(record),
            "manual_transcript_correction": "PENDING_in_UI_before_relying_on_transcript"}


def run_validation(*, model_kinds=("lr", "svm"), include_ui=False,
                   audio_path: Path | None = None, beto_artifact: Path | None = None) -> dict:
    """Measure executed contracts. Do not infer readiness from synthetic labels."""
    if not model_kinds or any(kind not in ("lr", "svm") for kind in model_kinds):
        raise ValueError("Solo se admiten los baselines DEMO lr y svm.")
    start, cpu_start = time.perf_counter(), time.process_time()
    report = {
        "schema": "traveler-functional-validation-v1",
        "scope": "EXECUTED_SOFTWARE_CONTRACTS_NOT_SCIENTIFIC_ACCURACY",
        "human_gold_reviews": 0,
        "fixtures": {"count": len(CASES), "origin": "artificial_functional_examples",
                     "historical_manual_negation_case": "retained_for_regression_only_not_tuning",
                     "evaluation_status": "NOT_A_NEW_INDEPENDENT_TEST_SET"},
        "environment": {"python": platform.python_version(), "platform": platform.system(),
                        "machine": platform.machine(), "logical_cpus": os.cpu_count()},
        "models": {}, "violations": [],
    }
    report["environment"]["packages"] = {}
    for name in ("scikit-learn", "numpy", "gradio"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "NOT_INSTALLED"
        report["environment"]["packages"][name] = version
    for kind in model_kinds:
        begin = time.perf_counter()
        model = train([x for x, _ in DEMO], [y for _, y in DEMO], model_kind=kind)
        trained_seconds = time.perf_counter() - begin
        cases = [_observed(case_id, text, model) for case_id, text in CASES]
        texts = parse_review_lines("\n".join(text for _, text in CASES))
        batch_start = time.perf_counter()
        batch = analyze_batch(texts, model, prefix="functional", model_kind=kind)
        batch_seconds = time.perf_counter() - batch_start
        corrected = apply_human_correction(batch, batch[0]["id"], "positivo", "Fixture de software, no dictamen humano.")
        json_bytes = export_json(corrected).encode("utf-8")
        csv_bytes = export_csv(corrected).encode("utf-8-sig")
        payload = json.loads(json_bytes)
        exported = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"))))
        if payload.get("n") != len(CASES) or len(exported) != len(CASES):
            report["violations"].append(kind + ":export_count_mismatch")
        if corrected[0]["sentimiento"] != batch[0]["sentimiento"] or batch[0]["etiqueta_corregida_humano"] is not None:
            report["violations"].append(kind + ":human_correction_overwrote_prediction")
        for row in cases:
            report["violations"].extend(kind + ":" + row["case_id"] + ":" + problem for problem in row["contract_violations"])
        for row in batch:
            report["violations"].extend(kind + ":batch:" + problem for problem in trace_violations(row))
        # Also exercise the actual UTF-8 CSV ingestion, including accents.
        sample = io.StringIO(newline="")
        writer = csv.writer(sample)
        writer.writerow(["reseña"])
        writer.writerows([[text] for text in texts])
        if parse_review_csv(sample.getvalue().encode("utf-8-sig")) != texts:
            report["violations"].append(kind + ":CSV_ingestion_changed_text")
        known = next(row for row in cases if row["case_id"] == "known_complex_negation")
        model_report = {
            "training_source": "existing_DEMO_only_unchanged",
            "training_rows": len(DEMO),
            "training_source_sha256": hashlib.sha256(json.dumps(DEMO, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
            "training_seconds": round(trained_seconds, 4), "cases": cases,
            "batch_rows": len(batch), "batch_seconds": round(batch_seconds, 4),
            "exports": {"json_roundtrip_rows": payload["n"], "csv_roundtrip_rows": len(exported),
                        "json_bytes": len(json_bytes), "csv_bytes": len(csv_bytes),
                        "human_correction": "SIMULATED_SOFTWARE_FIXTURE_NOT_HUMAN_GOLD"},
            "known_negation_observation": {"prediction": known["observed"]["sentimiento"],
                  "prior_qualitative_reading": "positivo", "is_still_a_limitation": known["observed"]["sentimiento"] != "positivo",
                  "note": "Historical manual regression case; no performance estimate or model selection."},
        }
        latencies = sorted(row["latency_ms"] for row in cases)
        model_report["inference_latency"] = {
            "n": len(latencies), "scope": "six_sequential_cold_and_warm_agent_calls_not_a_stable_benchmark",
            "median_ms": round((latencies[2] + latencies[3]) / 2, 3),
            "min_ms": latencies[0], "max_ms": latencies[-1]}
        if include_ui:
            try:
                model_report["ui"] = _ui_callbacks(kind)
                report["violations"].extend(kind + ":" + item for item in model_report["ui"]["violations"])
            except Exception as exc:
                # Do not publish private exception arguments or stack paths.
                model_report["ui"] = {"status": "FAILED", "error_type": type(exc).__name__}
                report["violations"].append(kind + ":UI_callback_execution_failed")
        else:
            model_report["ui"] = {"status": "NOT_EXECUTED_use_--ui"}
        if audio_path:
            try:
                model_report["audio"] = _real_audio(audio_path, model, kind)
                report["violations"].extend(kind + ":audio:" + item for item in model_report["audio"]["violations"])
            except Exception as exc:
                model_report["audio"] = {"status": "FAILED", "error_type": type(exc).__name__}
                report["violations"].append(kind + ":real_audio_execution_failed")
        else:
            model_report["audio"] = {"status": "PENDING_real_authorized_voice_input_not_supplied",
                "mocked_tests_are_real_voice_evidence": False}
        report["models"][kind] = model_report
    if beto_artifact is not None:
        previous = os.environ.get("TRAVELER_BETO_ARTIFACT")
        try:
            os.environ["TRAVELER_BETO_ARTIFACT"] = str(beto_artifact)
            try:
                report["beto_optional_ui"] = _ui_callbacks("beto")
                report["violations"].extend("beto:" + item for item in report["beto_optional_ui"]["violations"])
            except Exception as exc:
                report["beto_optional_ui"] = {"status":"FAILED", "error_type":type(exc).__name__}
                report["violations"].append("beto:UI_callback_execution_failed")
        finally:
            if previous is None: os.environ.pop("TRAVELER_BETO_ARTIFACT", None)
            else: os.environ["TRAVELER_BETO_ARTIFACT"] = previous
    report["resources"] = {**_resource_snapshot(), "total_wall_seconds": round(time.perf_counter() - start, 4),
                           "measured_cpu_seconds": round(time.process_time() - cpu_start, 4),
                           "gpu_used": False, "paid_services_used": False,
                           "monetary_cost": "NOT_MEASURED_local_host_cost_not_assumed_zero"}
    report["contract_status"] = "PASSED" if not report["violations"] else "FAILED"
    report["pending"] = ["Real authorized voice and editable-transcript check in Windows browser",
                         "Browser visual interaction/microphone permissions on presentation equipment",
                         "Human corpus admission and human classification performance",
                         "Requirements-based academic progress assessment; no 90-percent claim"]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validación funcional observada, no benchmark humano.")
    parser.add_argument("--models", nargs="+", choices=("lr", "svm"), default=["lr", "svm"])
    parser.add_argument("--ui", action="store_true", help="Ejecutar callbacks Gradio locales; no abre servidor.")
    parser.add_argument("--audio", type=Path, help="WAV PCM de voz real autorizado, opcional; sin descarga ASR automática.")
    parser.add_argument("--beto-artifact", type=Path, help="Artefacto BETO realmente entrenado; ejecuta sus callbacks e impide acreditar fallback como BETO.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists(): parser.error("Usar un informe nuevo; no sobrescribir evidencia anterior.")
    if args.audio:
        private = (Path.cwd() / "data/private").resolve()
        if not args.out.resolve().is_relative_to(private):
            parser.error("Con audio real, el informe debe permanecer bajo data/private.")
    previous = os.environ.get("TRAVELER_BETO_ARTIFACT")
    try:
        if args.beto_artifact: os.environ["TRAVELER_BETO_ARTIFACT"] = str(args.beto_artifact)
        report = run_validation(model_kinds=tuple(args.models), include_ui=args.ui, audio_path=args.audio,
                                beto_artifact=args.beto_artifact)
    finally:
        if previous is None: os.environ.pop("TRAVELER_BETO_ARTIFACT", None)
        else: os.environ["TRAVELER_BETO_ARTIFACT"] = previous
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["contract_status"], "models": list(report["models"]),
                      "violations": report["violations"], "audio": "EXECUTED" if args.audio else "PENDING_NO_REAL_INPUT"}, ensure_ascii=False))
    return 0 if report["contract_status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
