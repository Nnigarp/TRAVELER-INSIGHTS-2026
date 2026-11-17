"""Reproduce a small report battery; contracts by default, actual models only with --actual.

Not a predictive benchmark or proof of human-validated report quality. Unknown
syntax/copying remains observable in the report instead of being scored away.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BATTERY = ROOT / "tests/fixtures/qwen_report_quality_v2.json"


def inspect_content(record, case, *, actual=False):
    from traveler_agent.ollama_report import validate_report, ReportRejected
    problems = []
    if actual:
        from traveler_agent.demo_validation import trace_violations
        problems.extend(trace_violations(record))
        generation = record.get("generacion_qwen", {})
        if (generation.get("estado") != "GENERADO" or generation.get("generacion_llm") is not True or
            generation.get("validado_por") != "ActionAgent" or
            generation.get("modelo_ejecutado") != "qwen3:1.7b"):
            problems.append("QWEN_NOT_VALIDATED")
        report = generation.get("informe") or {}
        sentiment, aspects, review = record.get("sentimiento"), record.get("aspectos_exploratorios"), record.get("revision_humana")
        trace = record.get("multiagente", {}).get("traza_generacion", [])
        if len(trace) != 3 or trace[-1].get("status") != "OK":
            problems.append("GENERATION_TRACE_INVALID")
    else:
        report = record
        sentiment, aspects, review = case["sentiment"], case["aspects"], case["needs_human"]
    try:
        validate_report(report, case["text"], sentiment, aspects, review)
    except ReportRejected as exc:
        problems.append(str(exc))
    copied = report.get("resumen") == case["text"]
    if case["mode"] == "PARAFRASIS_CONTROLADA":
        if copied:
            problems.append("EXPECTED_SYNTHESIS_COPIES_SOURCE")
        if not all(term.casefold() in str(report.get("resumen", "")).casefold() for term in case["summary_terms"]):
            problems.append("EXPECTED_CONTENT_MISSING")
    observations = []
    if case["mode"] == "CITA_CONSERVADORA":
        observations.append("Sintaxis no cubierta: cita conservadora y solicitud de aclaración, sin paráfrasis del incidente.")
    if copied:
        observations.append("Resumen idéntico al texto original; no se acredita una mejora de síntesis en este caso.")
    return {"violations": problems, "summary_copies_source": copied, "observed_limitations": observations}


def run(*, actual=False, classifier="beto", gradio_url=None):
    from traveler_agent.ollama_report import report_schema, OllamaConfig
    battery = json.loads(BATTERY.read_text(encoding="utf-8"))
    rows, preflight = [], []
    identity = {"status": "NOT_EXECUTED_CONTRACT_MODE"}
    client = None
    if actual:
        from scripts.check_qwen_local import ollama_identity
        identity = ollama_identity()
        if identity.get("status") != "VERIFIED_LOCAL_API":
            preflight.append("OLLAMA_IDENTITY_NOT_VERIFIED")
        if gradio_url:
            OllamaConfig(endpoint=gradio_url)
            from gradio_client import Client
            client = Client(gradio_url, verbose=False, httpx_kwargs={"trust_env": False})
        else:
            import app
    for case in battery["cases"]:
        if actual:
            try:
                if client:
                    _, encoded, _ = client.predict(case["text"], classifier, True, api_name="/analyze_with_agents")
                else:
                    _, encoded, _ = app.analyze_with_agents(case["text"], classifier, True)
                record = json.loads(encoded)
                checked = inspect_content(record, case, actual=True)
                if record.get("modelo_utilizado") != classifier:
                    checked["violations"].append("CLASSIFIER_DID_NOT_EXECUTE")
                if classifier == "beto" and record.get("seleccion_modelo", {}).get("fallback") is not False:
                    checked["violations"].append("BETO_FALLBACK_NOT_ACCEPTED")
            except Exception as exc:
                record, checked = {}, {"violations": ["APPLICATION_REQUEST_FAILED", type(exc).__name__]}
        else:
            schema = report_schema(case["text"], case["sentiment"], case["aspects"], case["needs_human"])
            record = {key: spec["enum"][0] for key, spec in schema["properties"].items()}
            checked = inspect_content(record, case)
        rows.append({"id": case["id"], **checked, "observed": record})
    return {"scope": "ACTUAL_APPLICATION_REPORT_CHECK_NOT_PREDICTIVE_EVALUATION" if actual else
            "CONTRACT_ONLY_NO_LLM_OR_CLASSIFIER_EXECUTED",
            "battery_version": battery["version"], "ollama_identity": identity,
            "classifier_requested": classifier if actual else "NOT_EXECUTED",
            "preflight_violations": preflight, "cases": rows,
            "status": "FAILED" if preflight or any(r["violations"] for r in rows) else "PASSED",
            "limits": "Small synthetic cases with known contract expectations; no human quality labels, entailment proof or generalisation claim."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", action="store_true", help="Call the real local application/Ollama; default checks contracts only.")
    parser.add_argument("--classifier", choices=("lr", "svm", "beto"), default="beto")
    parser.add_argument("--gradio-url", help="Use an already running local Gradio server to reuse loaded models.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a new output filename; previous observations are preserved.")
    if args.gradio_url and not args.actual:
        parser.error("--gradio-url requires --actual")
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    try:
        report = run(actual=args.actual, classifier=args.classifier, gradio_url=args.gradio_url)
    except Exception as exc:
        report = {"status": "FAILED", "scope": "ACTUAL_REQUEST_FAILED" if args.actual else "CONTRACT_FAILED",
                  "failure_type": type(exc).__name__}
    report["utc"] = datetime.now(timezone.utc).isoformat()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    report["commit"] = commit.stdout.strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "scope")}, ensure_ascii=False))
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
