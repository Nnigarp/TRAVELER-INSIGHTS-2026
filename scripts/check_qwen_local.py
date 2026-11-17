"""Actual local Qwen + selected classifier acceptance. No mocked or fallback success."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ACCEPTANCE_TEXT = "El hotel era cómodo, pero el transporte nunca llegó y perdimos la excursión."


def check_record(record, requested):
    violations = []
    if record.get("modelo_utilizado") != requested:
        violations.append("CLASSIFIER_DID_NOT_EXECUTE")
    if requested == "beto" and record.get("seleccion_modelo", {}).get("fallback") is not False:
        violations.append("BETO_FALLBACK_NOT_ACCEPTED")
    generation = record.get("generacion_qwen", {})
    if generation.get("estado") != "GENERADO" or generation.get("validado_por") != "ActionAgent":
        violations.append("QWEN_REPORT_NOT_VALIDATED")
    if generation.get("modelo_ejecutado") != "qwen3:1.7b" or generation.get("modelo_solicitado") != "qwen3:1.7b":
        violations.append("QWEN_1_7B_DID_NOT_EXECUTE")
    if generation.get("acciones_externas_ejecutadas") is not False:
        violations.append("EXTERNAL_ACTION_FLAG_INVALID")
    report = generation.get("informe") or {}
    if "transporte" not in report.get("problema_identificado", "").casefold():
        violations.append("TRANSPORT_PROBLEM_NOT_IDENTIFIED")
    trace = record.get("multiagente", {}).get("traza", [])
    if len(trace) != 13 or [e.get("sequence") for e in trace] != list(range(1, 14)):
        violations.append("GENERATION_TRACE_INCOMPLETE")
    if not any(e.get("agent") == "accion" and e.get("task") == "generacion" and e.get("status") == "OK" for e in trace):
        violations.append("ACTION_REPORT_GATE_NOT_PASSED")
    return violations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", choices=("beto", "lr", "svm"), default="beto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Conserva la evidencia anterior; usa otro nombre de salida.")
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    import app
    _, encoded, _ = app.analyze_with_agents(ACCEPTANCE_TEXT, args.classifier, True)
    record = json.loads(encoded)
    violations = check_record(record, args.classifier)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    result = {"scope": "ACTUAL_LOCAL_OLLAMA_AND_CLASSIFIER_NOT_A_PREDICTIVE_BENCHMARK",
              "status": "PASSED" if not violations else "FAILED",
              "utc": datetime.now(timezone.utc).isoformat(),
              "commit": commit.stdout.strip(), "classifier_requested": args.classifier,
              "violations": violations, "observed": record}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "violations": violations,
                      "modelo_utilizado": record.get("modelo_utilizado"),
                      "generacion_qwen": {k: v for k, v in record.get("generacion_qwen", {}).items() if k != "informe"}},
                     ensure_ascii=False))
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
