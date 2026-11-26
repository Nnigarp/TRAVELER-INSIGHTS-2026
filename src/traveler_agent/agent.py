"""Public compatibility adapter for the hierarchical Traveler Insights agent.

Preserves the established inspect_review entrypoint and original JSON fields.
All decisions now pass through a stateful principal coordinator and four agents.
Qwen report generation and contextual verification are independently optional.
"""
from __future__ import annotations

import os

from .multiagent import TravelerOrchestrator


def inspect_review(text: str, model, *, review_id: str = "",
                   threshold: float = 0.56, modality: str = "texto",
                   edited_by_human: bool = False, qwen_enabled: bool = False,
                   reporter=None, context_verifier=None, progress=None) -> dict:
    """Run the hierarchical workflow while retaining the legacy API.

    - qwen_enabled / reporter → optional generative report after Control.
    - context_verifier → optional balance check before Control (does not write labels).
    When the classifier opts into context_verification and no verifier is passed,
    a default Ollama verifier is created independently of the report checkbox,
    unless TRAVELER_QWEN_VERIFY=0.
    """
    if qwen_enabled and reporter is None:
        from .ollama_report import OllamaReporter
        reporter = OllamaReporter()
    verify_flag = getattr(model, "context_verification", False)
    if context_verifier is None and verify_flag and os.getenv("TRAVELER_QWEN_VERIFY", "1") != "0":
        from .ollama_report import OllamaReporter
        context_verifier = OllamaReporter()
    return TravelerOrchestrator(
        model, threshold=threshold,
        reporter=reporter if qwen_enabled else None,
        context_verifier=context_verifier,
    ).run(
        text, review_id=review_id, modality=modality,
        edited_by_human=edited_by_human, progress=progress,
    )
