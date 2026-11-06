"""Public compatibility adapter for the hierarchical Traveler Insights agent.

Preserves the established inspect_review entrypoint and original JSON fields.
All decisions now pass through a stateful principal coordinator and four agents.
"""
from .multiagent import TravelerOrchestrator


def inspect_review(text: str, model, *, review_id: str = "",
                   threshold: float = 0.56, modality: str = "texto",
                   edited_by_human: bool = False) -> dict:
    """Run the hierarchical workflow while retaining the legacy API."""
    return TravelerOrchestrator(model, threshold=threshold).run(
        text, review_id=review_id, modality=modality,
        edited_by_human=edited_by_human,
    )
