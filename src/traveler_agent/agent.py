"""Public compatibility adapter for the hierarchical Traveler Insights agent.

Preserves the established inspect_review entrypoint and original JSON fields.
All decisions now pass through a stateful principal coordinator and four agents.
"""
from .multiagent import TravelerOrchestrator


def inspect_review(text: str, model, *, review_id: str = "",
                   threshold: float = 0.56, modality: str = "texto",
                   edited_by_human: bool = False, qwen_enabled: bool = False, reporter=None) -> dict:
    """Run the hierarchical workflow while retaining the legacy API."""
    if qwen_enabled and reporter is None:
        from .ollama_report import OllamaReporter
        reporter = OllamaReporter()
    return TravelerOrchestrator(model, threshold=threshold, reporter=reporter if qwen_enabled else None).run(
        text, review_id=review_id, modality=modality,
        edited_by_human=edited_by_human,
    )
