"""Hermetic pytest defaults: host V4/Ollama env must not alter classical contracts."""
from __future__ import annotations

import pytest

import app


@pytest.fixture(autouse=True)
def _isolate_traveler_host_env(monkeypatch):
    """Clear academic V4 selection leakage from the developer shell.

    When TRAVELER_BETO_ARTIFACT points at V4.2 partial256, get_demo_model() swaps
    classical LR/SVM for embedding heads with context_verification. That is valid
    for the academic demo, but it must not silently rewrite the default test
    contracts (UI labels, seleccion_modelo, Ollama call counts, simulated schemas).
    Tests that need the artifact re-enable it explicitly via monkeypatch.setenv.
    """
    monkeypatch.delenv("TRAVELER_BETO_ARTIFACT", raising=False)
    clearer = getattr(app.get_demo_model, "cache_clear", None)
    if callable(clearer):
        clearer()
    yield
    clearer = getattr(app.get_demo_model, "cache_clear", None)
    if callable(clearer):
        clearer()
