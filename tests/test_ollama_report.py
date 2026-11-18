"""Local HTTP API simulations; no Ollama executable, weights or downloads."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time

import pytest

from traveler_agent.ollama_report import (
    DEFAULT_MODEL, OllamaConfig, OllamaReporter, ReportRejected,
    render_generation_html, validate_generation, validate_report, report_schema,
)

TEXT = "El hotel era cómodo, pero el transporte nunca llegó y perdimos la excursión."


def fixture_report(text=TEXT, sentiment="negativo", needs_human=True, aspects=None):
    # API fixture, deliberately NOT evidence of actual Qwen inference.
    schema = report_schema(text, sentiment, aspects or [], needs_human)
    return {field: spec["enum"][0] for field, spec in schema["properties"].items()}


def response_for(report=None, **overrides):
    return {"model": DEFAULT_MODEL, "done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(report or fixture_report(), ensure_ascii=False)},
            **overrides}


@contextmanager
def simulated_ollama(reply=None, *, status=200, delay=0, drip=0):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Diagnostic fixtures, not a real Ollama daemon/model digest.
            value = {"version": "SIMULATED_TEST_ONLY"} if self.path == "/api/version" else {
                "models": [{"name": DEFAULT_MODEL, "digest": "0" * 64}]}
            raw = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def do_POST(self):
            requests.append({"path": self.path, "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                             "content_type": self.headers.get("Content-Type")})
            if delay:
                time.sleep(delay)
            value = reply(requests[-1]["body"]) if callable(reply) else reply
            raw = value if isinstance(value, bytes) else json.dumps(value if value is not None else response_for()).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                if drip:
                    for byte in raw:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(drip)
                else:
                    self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # Expected when the client times out.
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_real_http_structured_generation_and_separate_thinking(monkeypatch):
    reply = response_for(message={"role": "assistant", "thinking": "PRIVATE_REASONING_NEVER_RETAIN",
                                  "content": '<think>PRIVATE_INLINE</think>' + json.dumps(fixture_report())})
    with simulated_ollama(reply) as (url, calls):
        # Verify local reviews are not routed through proxy environment variables.
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
        result = OllamaReporter(OllamaConfig(url)).generate(TEXT, "negativo", True, [])
    assert result["estado"] == "GENERADO"
    assert result["modelo_ejecutado"] == DEFAULT_MODEL
    assert result["latencia_ms"] >= 0
    assert result["acciones_externas_ejecutadas"] is False
    assert "PRIVATE" not in json.dumps(result)
    assert len(calls) == 1 and calls[0]["path"] == "/api/chat"
    payload = calls[0]["body"]
    assert payload["think"] is False and payload["stream"] is False
    assert payload["options"]["temperature"] == 0 and payload["options"]["num_predict"] == 1024
    assert set(payload["format"]["required"]) == set(fixture_report())
    assert payload["format"]["additionalProperties"] is False
    assert "tools" not in payload
    assert json.loads(payload["messages"][1]["content"])["resena_no_confiable"] == TEXT
    assert validate_generation(result, TEXT, "negativo", [], True)["validado_por"] == "ActionAgent"


@pytest.mark.parametrize("reply,code", [
    (b"not json PRIVATE", "JSON_INVALIDO"),
    (response_for(message={"role": "assistant", "content": "not JSON"}), "JSON_INVALIDO"),
    (response_for(report={"resumen": "missing fields"}), "CAMPOS_INVALIDOS"),
    (response_for(report={**fixture_report(), "sentimiento": "positivo"}), "CAMPOS_INVALIDOS"),
    (response_for(report={**fixture_report(), "revision_humana": False}), "CAMPOS_INVALIDOS"),
    (response_for(report={**fixture_report(), "resumen": 2}), "CAMPOS_INVALIDOS"),
    (response_for(report={**fixture_report(), "recomendacion": " "}), "CAMPOS_INVALIDOS"),
    (response_for(report={**fixture_report(), "resumen": "a"*801}), "CAMPOS_INVALIDOS"),
    (response_for(message={"role": "assistant", "content": '<think>PRIVATE'}), "RAZONAMIENTO_SIN_RESPUESTA"),
    (response_for(message={"role": "assistant", "content": '{"resumen":"a","resumen":"b"}'}), "JSON_INVALIDO"),
    (response_for(done=False), "GENERACION_INCOMPLETA"),
    (response_for(done_reason="length"), "GENERACION_INCOMPLETA"),
    (response_for(model="unexpected-model"), "MODELO_INESPERADO"),
    (response_for(message={"role": "assistant", "content": "{}", "tool_calls": [{"name": "send"}]}), "RESPUESTA_INVALIDA"),
    (response_for(message={"role": "system", "content": "{}"}), "RESPUESTA_INVALIDA"),
    (b"x"*65537, "RESPUESTA_EXCESIVA"),
    ([], "RESPUESTA_INVALIDA"),
])
def test_invalid_response_never_publishes_an_invented_report(reply, code):
    with simulated_ollama(reply) as (url, calls):
        result = OllamaReporter(OllamaConfig(url)).generate(TEXT, "negativo", True, [])
    assert result["estado"] == "ERROR" and result["codigo"] == code
    assert result["informe"] is None
    assert "PRIVATE" not in json.dumps(result)
    assert len(calls) == 1  # No silent retry or model substitution.


@pytest.mark.parametrize("status,code", [(404, "MODELO_NO_DISPONIBLE"), (500, "OLLAMA_HTTP_ERROR"), (400, "OLLAMA_HTTP_ERROR")])
def test_http_failures(status, code):
    with simulated_ollama({"error": "PRIVATE"}, status=status) as (url, calls):
        result = OllamaReporter(OllamaConfig(url)).generate(TEXT, "negativo", True, [])
    assert result["codigo"] == code and result["modelo_ejecutado"] is None
    assert result["informe"] is None and len(calls) == 1
    assert "PRIVATE" not in json.dumps(result)


def test_timeout_and_disconnected():
    with simulated_ollama(delay=.2) as (url, _):
        result = OllamaReporter(OllamaConfig(url, timeout=.02)).generate(TEXT, "negativo", True, [])
    assert result["codigo"] == "TIMEOUT" and result["informe"] is None
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{probe.getsockname()[1]}"
    result = OllamaReporter(OllamaConfig(url)).generate(TEXT, "negativo", True, [])
    assert result["codigo"] == "OLLAMA_DESCONECTADO" and result["modelo_ejecutado"] is None


@pytest.mark.parametrize("field,value", [
    ("resumen", "El hotel canceló el servicio."),
    ("problema_identificado", "El hotel es culpable."),
    ("recomendacion", "Verificar el reembolso aprobado."),
    ("accion_propuesta", "Documentar una devolución de dinero."),
    ("accion_propuesta", "Verificar la salida el viernes."),
    ("accion_propuesta", "Solicitar 200 euros."),
    ("accion_propuesta", "Ya contactamos al hotel."),
    ("accion_propuesta", "Consultar al responsable culpable."),
    ("accion_propuesta", "Revisar el correo enviado al hotel."),
])
def test_action_gate_rejects_ungrounded_facts_and_execution_claims(field, value):
    report = {**fixture_report(), field: value}
    with pytest.raises(ReportRejected):
        validate_report(report, TEXT, "negativo", [], True)
    record = validate_generation({"estado": "GENERADO", "informe": report}, TEXT, "negativo", [], True)
    assert record["estado"] == "ERROR" and record["informe"] is None


def test_configuration_and_safe_html(monkeypatch):
    for endpoint in ("https://example.com", "file:///tmp", "http://user@localhost", "http://localhost:11434/api/chat"):
        with pytest.raises(ValueError):
            OllamaConfig(endpoint)
    for timeout in (0, float("nan"), 601):
        with pytest.raises(ValueError):
            OllamaConfig(timeout=timeout)
    monkeypatch.setenv("TRAVELER_OLLAMA_TIMEOUT", "bad")
    result = OllamaReporter().generate(TEXT, "neutral", False, [])
    assert result["codigo"] == "CONFIGURACION_INVALIDA"
    source = '<script>alert("x")</script>\r\n\r\n![image](https://example.com)\n[link](https://example.com)'
    generation = validate_generation({"estado": "GENERADO", "codigo": "OK", "modelo_solicitado": DEFAULT_MODEL,
        "modelo_ejecutado": DEFAULT_MODEL, "latencia_ms": 0, "informe": fixture_report(source)}, source, "negativo", [], True)
    html = render_generation_html({"generacion_qwen": generation})
    assert '<script>' not in html and "&lt;script&gt;" in html
    assert '<a ' not in html and '<img' not in html
    assert "\n\n" not in html and "\r" not in html
    assert "<br><br>&#33;&#91;image&#93;" in html


@pytest.mark.parametrize("field,value", [
    ("sentimiento_analizado", "positivo"),
    ("requiere_revision_humana", False),
    ("aspectos_identificados", ["seguridad"]),
    ("requiere_revision_humana", 1),
    ("aspectos_identificados", "acceso"),
    ("advertencias", []),
    ("recomendacion", "Verificar el pago autorizado al hotel."),
    ("accion_propuesta", "Documentar que el hotel pagará la excursión."),
])
def test_locked_fields_and_controlled_proposals(field, value):
    with simulated_ollama(response_for({**fixture_report(), field: value})) as (url, _):
        result = OllamaReporter(OllamaConfig(url)).generate(TEXT, "negativo", True, [])
    assert result["generacion_llm"] is False and result["informe"] is None


def test_complete_cites_do_not_drop_negation_or_contrast():
    source = "No confirmo que el hotel cancelara el taxi, pero perdimos la visita."
    report = fixture_report(source)
    report["problema_identificado"] = "el hotel cancelara el taxi"
    assert report["problema_identificado"] in source  # The old substring gate accepted this.
    with pytest.raises(ReportRejected, match="HECHOS_NO_SUSTENTADOS"):
        validate_report(report, source, "negativo", [], True)


def test_deadline_stops_dripping_body_not_only_socket_inactivity():
    started = time.monotonic()
    with simulated_ollama(drip=.01) as (url, _):
        result = OllamaReporter(OllamaConfig(url, timeout=.07)).generate(TEXT, "negativo", True, [])
    assert result["codigo"] == "TIMEOUT"
    assert time.monotonic() - started < 2


def test_redirects_are_not_followed():
    with simulated_ollama(status=302) as (url, calls):
        result = OllamaReporter(OllamaConfig(url)).generate(TEXT, "negativo", True, [])
    assert result["codigo"] == "OLLAMA_HTTP_ERROR" and len(calls) == 1


def test_long_input_is_not_silently_cut_and_does_not_make_http_calls():
    with simulated_ollama() as (url, calls):
        result = OllamaReporter(OllamaConfig(url)).generate("Una visita. " * 150, "neutral", False, [])
    assert result["codigo"] == "ENTRADA_GENERATIVA_EXCESIVA" and calls == []


def test_failed_envelope_cannot_retain_a_partial_report_or_true_generation_flag():
    failed = {"estado": "ERROR", "codigo": "TIMEOUT", "generacion_llm": True, "informe": fixture_report()}
    result = validate_generation(failed, TEXT, "negativo", [], True)
    assert result["generacion_llm"] is False and result["informe"] is None
    assert result["alternativa"] == "RECOMENDACION_TRADICIONAL_POR_REGLAS"
