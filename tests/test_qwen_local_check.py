"""The Windows acceptance command never accepts simulated fallback as BETO/Qwen."""
import copy
import importlib.util
from pathlib import Path

from traveler_agent.multiagent import TravelerOrchestrator
from traveler_agent.ollama_report import OllamaConfig, OllamaReporter
import app
from test_ollama_report import TEXT, simulated_ollama
from test_qwen_integration import from_request

spec = importlib.util.spec_from_file_location("check_qwen_local", Path(__file__).resolve().parents[1] / "scripts/check_qwen_local.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_acceptance_checker_requires_real_model_names_no_fallback():
    with simulated_ollama(from_request) as (url, _):
        record = TravelerOrchestrator(app.get_demo_model(), reporter=OllamaReporter(OllamaConfig(url))).run(TEXT)
    record["modelo_utilizado"] = "lr"
    assert check.check_record(record, "lr") == []
    assert "CLASSIFIER_DID_NOT_EXECUTE" in check.check_record(record, "beto")
    assert "BETO_FALLBACK_NOT_ACCEPTED" in check.check_record(record, "beto")
    broken = copy.deepcopy(record)
    broken["generacion_qwen"].update(estado="ERROR", informe=None)
    assert "QWEN_REPORT_NOT_VALIDATED" in check.check_record(broken, "lr")
    broken = copy.deepcopy(record)
    broken["generacion_qwen"]["modelo_ejecutado"] = "unexpected-model"
    assert "QWEN_1_7B_DID_NOT_EXECUTE" in check.check_record(broken, "lr")
    broken = copy.deepcopy(record)
    broken["multiagente"]["traza"] = record["multiagente"]["traza"][:10]
    assert "GENERATION_TRACE_INCOMPLETE" in check.check_record(broken, "lr")
