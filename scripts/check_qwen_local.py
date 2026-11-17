"""Local Qwen + selected classifier acceptance; no mocked or fallback success."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ACCEPTANCE_TEXT = "El hotel era cómodo, pero el transporte nunca llegó y perdimos la excursión."


def check_record(record, requested):
    from traveler_agent.demo_validation import trace_violations
    from traveler_agent.ollama_report import validate_report, ReportRejected
    violations = trace_violations(record)
    if record.get("modelo_utilizado") != requested:
        violations.append("CLASSIFIER_DID_NOT_EXECUTE")
    selected = record.get("seleccion_modelo", {})
    if requested == "beto" and (selected.get("fallback") is not False or
                                selected.get("effective") != "beto"):
        violations.append("BETO_FALLBACK_NOT_ACCEPTED")
    generation = record.get("generacion_qwen", {})
    if (generation.get("estado") != "GENERADO" or generation.get("validado_por") != "ActionAgent" or
        generation.get("generacion_llm") is not True):
        violations.append("QWEN_REPORT_NOT_VALIDATED")
    if generation.get("modelo_ejecutado") != "qwen3:1.7b" or generation.get("modelo_solicitado") != "qwen3:1.7b":
        violations.append("QWEN_1_7B_DID_NOT_EXECUTE")
    if generation.get("acciones_externas_ejecutadas") is not False:
        violations.append("EXTERNAL_ACTION_FLAG_INVALID")
    report = generation.get("informe") or {}
    try:
        validate_report(report, ACCEPTANCE_TEXT, record.get("sentimiento"),
                        record.get("aspectos_exploratorios"), record.get("revision_humana"))
    except ReportRejected:
        violations.append("REPORT_INCONSISTENT_WITH_EVIDENCE")
    if "transporte" not in str(report.get("problema_identificado", "")).casefold():
        violations.append("TRANSPORT_PROBLEM_NOT_IDENTIFIED")
    trace = record.get("multiagente", {}).get("traza_generacion", [])
    if len(trace) != 3 or [e.get("sequence") for e in trace] != [1, 2, 3]:
        violations.append("GENERATION_TRACE_INCOMPLETE")
    if not any(e.get("agent") == "accion" and e.get("status") == "OK" for e in trace):
        violations.append("ACTION_REPORT_GATE_NOT_PASSED")
    return violations


def resource_snapshot():
    """Current RSS snapshots, not isolated model memory or peak GPU/RAM."""
    try:
        import psutil
        processes = []
        for process in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                name = (process.info["name"] or "").lower()
                if "ollama" in name:
                    processes.append({"pid": process.pid, "name": process.info["name"],
                                      "rss_mib": round(process.info["memory_info"].rss / 1048576, 2)})
            except (psutil.Error, AttributeError):
                continue
        memory = psutil.virtual_memory()
        return {"status": "MEASURED_CURRENT_SNAPSHOT", "python_check_rss_mib": round(
                    psutil.Process().memory_info().rss / 1048576, 2),
                "system_available_mib": round(memory.available / 1048576, 2),
                "system_total_mib": round(memory.total / 1048576, 2),
                "ollama_processes": processes,
                "limits": "Current RSS, not peak; shared pages may overlap. Python check RSS is not Gradio server RSS in HTTP mode."}
    except ImportError:
        return {"status": "PENDING_OPTIONAL_PSUTIL", "limits": "Use Windows Task Manager; no RAM estimate invented."}


def ollama_identity():
    from traveler_agent.ollama_report import OllamaConfig, local_json_request, ReportRejected
    try:
        config = OllamaConfig.from_env()
        version = local_json_request(config, "/api/version", timeout=5)
        tags = local_json_request(config, "/api/tags", timeout=5)
        match = next((m for m in tags.get("models", []) if isinstance(m, dict) and
                      (m.get("name") == config.model or m.get("model") == config.model)), {})
        digest = match.get("digest")
        if not isinstance(version.get("version"), str) or not isinstance(digest, str) or not re.fullmatch(r"(?:sha256:)?[a-f0-9]{64}", digest):
            return {"status": "IDENTITY_NOT_VERIFIED"}
        return {"status": "VERIFIED_LOCAL_API", "ollama_version": version["version"],
                "model": config.model, "digest": digest,
                "limits": "Local API reports this digest; no remote attestation or bitwise determinism guarantee."}
    except (ReportRejected, ValueError, TypeError, AttributeError):
        return {"status": "IDENTITY_UNAVAILABLE"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", choices=("beto", "lr", "svm"), default="beto")
    parser.add_argument("--gradio-url", help="Existing local Gradio server; omit to test application callbacks directly.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Conserva la evidencia anterior; usa otro nombre de salida.")
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    before = resource_snapshot()
    identity = ollama_identity()
    started = time.perf_counter()
    try:
        if args.gradio_url:
            from traveler_agent.ollama_report import OllamaConfig
            OllamaConfig(endpoint=args.gradio_url)  # Same loopback-only restriction.
            from gradio_client import Client
            client = Client(args.gradio_url, verbose=False, httpx_kwargs={"trust_env": False})
            _, encoded, _ = client.predict(ACCEPTANCE_TEXT, args.classifier, True, api_name="/analyze_with_agents")
        else:
            import app
            _, encoded, _ = app.analyze_with_agents(ACCEPTANCE_TEXT, args.classifier, True)
        record = json.loads(encoded)
        violations = check_record(record, args.classifier)
    except Exception:
        record, violations = {}, ["APPLICATION_REQUEST_FAILED"]
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    if identity["status"] != "VERIFIED_LOCAL_API":
        violations.append("OLLAMA_IDENTITY_NOT_VERIFIED")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    result = {"scope": "LOCAL_OLLAMA_APPLICATION_ACCEPTANCE_NOT_A_PREDICTIVE_BENCHMARK",
              "transport": "GRADIO_HTTP" if args.gradio_url else "APPLICATION_CALLBACK",
              "status": "PASSED" if not violations else "FAILED", "utc": datetime.now(timezone.utc).isoformat(),
              "commit": commit.stdout.strip(), "python": platform.python_version(), "os": platform.system(),
              "classifier_requested": args.classifier, "application_elapsed_ms": elapsed,
              "ollama_identity": identity, "resources_before": before, "resources_after": resource_snapshot(),
              "violations": violations, "observed": record}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves evidence even if another process races us.
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "violations": violations,
                      "modelo_utilizado": record.get("modelo_utilizado"), "output": str(args.output)}, ensure_ascii=False))
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
