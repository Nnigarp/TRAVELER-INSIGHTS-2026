"""Actual Gradio HTTP/queue/session/downloads with a simulated local Ollama API."""
import csv
import io
import json
from pathlib import Path
import socket

from gradio_client import Client, handle_file

import app
from test_ollama_report import TEXT, simulated_ollama
from test_qwen_integration import from_request


def test_gradio_qwen_http_and_existing_default_clients(tmp_path, monkeypatch):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with simulated_ollama(from_request) as (url, calls):
        monkeypatch.setenv("TRAVELER_OLLAMA_ENDPOINT", url)
        try:
            app.app.queue(default_concurrency_limit=1).launch(
                server_name="127.0.0.1", server_port=port, share=False, inbrowser=False,
                prevent_thread_lock=True, quiet=True,
                **({"theme": app.THEME, "css": app.UI_CSS} if app._GRADIO_MAJOR >= 6 else {}))
            client = Client(f"http://127.0.0.1:{port}", verbose=False, httpx_kwargs={"trust_env": False})
            # Old two-argument clients must still work, and never call Qwen.
            _, encoded, _ = client.predict(TEXT, "lr", api_name="/analyze_with_agents")
            assert "generacion_qwen" not in json.loads(encoded) and len(calls) == 0
            for kind in ("lr", "svm"):
                summary, encoded, view = client.predict(TEXT, kind, True, api_name="/analyze_with_agents")
                record = json.loads(encoded)
                assert record["modelo_utilizado"] == kind
                assert record["generacion_qwen"]["modelo_ejecutado"] == "qwen3:1.7b"
                assert record["generacion_qwen"]["validado_por"] == "ActionAgent"
                assert "Problema" in summary and "sin ejecutar" in summary
                assert "Qwen local" in view and len(record["multiagente"]["traza"]) == 10 and len(record["multiagente"]["traza_generacion"]) == 3
            batch = client.predict(TEXT, "lr", True, api_name="/run_batch_lines")
            assert "1/1" in batch[-1]
            client.predict("demo-001", "negativo", "Revisión", api_name="/revise")
            json_path, csv_path, _ = client.predict(api_name="/export_files")
            rows = json.loads(Path(json_path).read_text(encoding="utf-8"))["registros"]
            assert rows[0]["etiqueta_corregida_humano"] == "negativo"
            assert rows[0]["generacion_qwen"]["estado"] == "GENERADO"
            csv_rows = list(csv.DictReader(io.StringIO(Path(csv_path).read_text(encoding="utf-8-sig"))))
            assert csv_rows[0]["qwen_modelo_ejecutado"] == "qwen3:1.7b"
            source = tmp_path / "upload.csv"
            source.write_text('reseña\n"' + TEXT + '"\n', encoding="utf-8")
            result = client.predict(handle_file(str(source)), "svm", True, api_name="/run_batch_csv")
            assert "1/1" in result[-1]
            from test_qwen_local_check import check
            evidence_path = tmp_path / "simulated-api-acceptance.json"
            assert check.main(["--classifier", "lr", "--gradio-url", f"http://127.0.0.1:{port}",
                               "--output", str(evidence_path)]) == 0
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            assert evidence["transport"] == "GRADIO_HTTP"
            assert evidence["ollama_identity"]["ollama_version"] == "SIMULATED_TEST_ONLY"
            assert evidence["observed"]["generacion_qwen"]["generacion_llm"] is True
            # Explicit connection failure remains successful classifier HTTP response.
            monkeypatch.setenv("TRAVELER_OLLAMA_ENDPOINT", "http://127.0.0.1:1")
            summary, encoded, _ = client.predict(TEXT, "svm", True, api_name="/analyze_with_agents")
            record = json.loads(encoded)
            assert record["modelo_utilizado"] == "svm"
            assert record["generacion_qwen"]["estado"] == "ERROR"
            assert record["generacion_qwen"]["informe"] is None
            assert "análisis convencional" in summary
        finally:
            app.app.close()
    assert len(calls) == 5
