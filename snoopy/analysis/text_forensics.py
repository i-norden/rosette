"""Text-based forensics for detecting paper mill artifacts.

Implements tortured phrase detection — a high-value signal for identifying
papers produced by paper mills that use synonym substitution to evade
plagiarism detectors.  E.g. "artificial neural network" becomes "fake neural
system".

Reference: Cabanac, Labbé & Magazinov (2021). "Tortured phrases: A dubious
writing style emerging in science."
"""

from __future__ import annotations

import functools
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the curated tortured phrase dictionary
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_PHRASES_PATH = _DATA_DIR / "tortured_phrases.json"


@functools.lru_cache(maxsize=1)
def _load_phrases() -> dict[str, str]:
    """Load the tortured phrase dictionary (lazy, cached, thread-safe via lru_cache).

    Returns:
        Dict mapping tortured phrase (lower-case) to its correct equivalent.
    """
    if _PHRASES_PATH.exists():
        try:
            with open(_PHRASES_PATH) as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                return {k.lower(): v for k, v in raw.items()}
            elif isinstance(raw, list):
                # List of {"tortured": ..., "correct": ...} dicts
                return {
                    entry["tortured"].lower(): entry.get("correct", "")
                    for entry in raw
                    if isinstance(entry, dict) and "tortured" in entry
                }
            else:
                return {}
        except Exception as e:
            logger.warning("Failed to load tortured phrases: %s", e)
            return {}
    else:
        logger.debug("Tortured phrases file not found at %s", _PHRASES_PATH)
        return {}


@dataclass
class TorturedPhraseMatch:
    """A single tortured phrase match in the text."""

    tortured_phrase: str
    correct_phrase: str
    position: int
    context: str


@dataclass
class TorturedPhraseResult:
    """Result of tortured phrase detection."""

    suspicious: bool
    matches: list[TorturedPhraseMatch] = field(default_factory=list)
    match_count: int = 0
    unique_phrases: int = 0
    details: str = ""


def detect_tortured_phrases(
    text: str,
    min_matches: int = 2,
) -> TorturedPhraseResult:
    """Search text for known tortured phrases from paper mills.

    Tortured phrases arise when paper mills use synonym substitution to evade
    plagiarism detection.  Examples:
    - "artificial neural network" → "fake neural system"
    - "deep learning" → "profound learning"
    - "random forest" → "arbitrary woodland"

    Args:
        text: The full paper text to search.
        min_matches: Minimum number of distinct tortured phrase matches to
            flag the paper as suspicious.

    Returns:
        TorturedPhraseResult with all matches found.
    """
    phrases = _load_phrases()
    if not phrases or not text:
        return TorturedPhraseResult(suspicious=False, details="No phrases to check")

    text_lower = text.lower()
    matches: list[TorturedPhraseMatch] = []
    seen_phrases: set[str] = set()

    for tortured, correct in phrases.items():
        # Use word-boundary matching to avoid false positives on substrings
        pattern = r"\b" + re.escape(tortured) + r"\b"
        for m in re.finditer(pattern, text_lower):
            pos = m.start()
            # Extract surrounding context
            ctx_start = max(0, pos - 40)
            ctx_end = min(len(text), pos + len(tortured) + 40)
            context = text[ctx_start:ctx_end].strip()

            matches.append(
                TorturedPhraseMatch(
                    tortured_phrase=tortured,
                    correct_phrase=correct,
                    position=pos,
                    context=context,
                )
            )
            seen_phrases.add(tortured)

    unique_count = len(seen_phrases)
    suspicious = unique_count >= min_matches

    if matches:
        examples = [f"'{m.tortured_phrase}' (should be '{m.correct_phrase}')" for m in matches[:5]]
        details = (
            f"Found {len(matches)} tortured phrase occurrence(s) "
            f"({unique_count} unique): {'; '.join(examples)}"
        )
    else:
        details = "No tortured phrases detected"

    return TorturedPhraseResult(
        suspicious=suspicious,
        matches=matches,
        match_count=len(matches),
        unique_phrases=unique_count,
        details=details,
    )
