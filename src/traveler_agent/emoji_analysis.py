"""Exploratory, privacy-preserving Unicode emoji markers for dataset profiling.

This is NOT an exhaustive Unicode grapheme segmenter or sentiment classifier.
Markers represent code points (not full emoji sequences); do not infer polarity.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Any

# Broad Unicode blocks used for exploratory analysis. Does not cover every emoji.
_RANGES = (
    (0x1F300, 0x1FAFF),  # pictographs and symbols
    (0x2600, 0x27BF),    # miscellaneous symbols/dingbats
    (0x1F1E6, 0x1F1FF),  # flag regional indicators
)
_EMOTICON = re.compile(r"(?<!\w)(?:[:;=8][-^']?[)(DPp/\\|]|[)(][-^']?[:;=8])(?!\w)")


def emoji_marker_count(text: str) -> int:
    """Count indicative Unicode code points, NOT grapheme-cluster emoji."""
    return sum(any(lo <= ord(ch) <= hi for lo, hi in _RANGES) for ch in text)


def has_emoticon(text: str) -> bool:
    return bool(_EMOTICON.search(text))


def emoji_profile(texts: Iterable[Any], labels: Iterable[Any] | None = None) -> dict:
    """Aggregate only: never expose review strings, authors, or exact emoji examples."""
    texts = list(texts)
    label_list = list(labels) if labels is not None else None
    if label_list is not None and len(texts) != len(label_list):
        raise ValueError("Text/label row counts differ")
    counts = Counter()
    for i, value in enumerate(texts):
        s = value if isinstance(value, str) else ""
        n = emoji_marker_count(s)
        present = n > 0
        emoticon = has_emoticon(s)
        counts["rows"] += 1
        counts["emoji_marker_codepoints"] += n
        counts["rows_with_emoji_markers"] += int(present)
        counts["rows_with_ascii_emoticons"] += int(emoticon)
        if label_list is not None:
            label = str(label_list[i])
            # Label values are descriptive only; do not assume numeric mapping.
            counts[("label", label, "rows")] += 1
            counts[("label", label, "with_emoji")] += int(present)
    grouped = []
    if label_list is not None:
        for label in sorted({str(v) for v in label_list}):
            total = counts[("label", label, "rows")]
            with_emoji = counts[("label", label, "with_emoji")]
            grouped.append({"label_raw": label, "rows": total,
                            "rows_with_emoji_markers": with_emoji})
    return {
        "method": "unicode_block_codepoint_heuristic_v1",
        "rows": counts["rows"],
        "rows_with_emoji_markers": counts["rows_with_emoji_markers"],
        "emoji_marker_codepoints": counts["emoji_marker_codepoints"],
        "rows_with_ascii_emoticons": counts["rows_with_ascii_emoticons"],
        "raw_label_groups": grouped,
        "limitations": [
            "Not full Unicode emoji coverage; composed sequences can count as multiple markers.",
            "Emojis/emoticons do not imply a fixed sentiment without linguistic context.",
            "Do not infer numerical sentiment label meanings from these counts."
        ]
    }
