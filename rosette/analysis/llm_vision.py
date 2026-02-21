"""LLM-based image analysis using a vision-capable LLM provider.

Provides a two-stage approach: a fast screening pass to flag potentially
manipulated figures, followed by a detailed analysis for flagged images.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from rosette.llm.prompts import (
    PROMPT_ANALYZE_FIGURE,
    PROMPT_FIGURE_CLASSIFY,
    PROMPT_SCREEN_FIGURE,
)

logger = logging.getLogger(__name__)


@dataclass
class VisionFinding:
    """A single finding from detailed vision analysis."""

    finding_type: str
    description: str
    location: str
    confidence: float


@dataclass
class ScreeningResult:
    """Result of the fast screening pass."""

    suspicious: bool
    reason: str
    confidence: float
    model_used: str
    raw_response: str
    parse_failed: bool = False


@dataclass
class DetailedAnalysisResult:
    """Result of the detailed analysis pass."""

    findings: list[VisionFinding]
    overall_assessment: str
    manipulation_likelihood: float
    model_used: str
    raw_response: str
    parse_failed: bool = False


def _parse_json_response(raw: str) -> dict:
    """Attempt to extract a JSON object from a model response string.

    The model may wrap the JSON in markdown code fences or include extra text.
    This helper tries several strategies to extract valid JSON.
    """
    # Try parsing directly first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    for marker in ("```json", "```"):
        if marker in raw:
            start = raw.index(marker) + len(marker)
            end = raw.index("```", start) if "```" in raw[start:] else len(raw)
            try:
                return json.loads(raw[start:end].strip())
            except json.JSONDecodeError:
                pass

    # Try finding first { ... } or [ ... ] block
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = raw.find(open_ch)
        end = raw.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass

    logger.warning("Failed to parse JSON from model response: %.200s", raw[:200])
    return {"_parse_failed": True}


async def screen_figure(
    image_path: str,
    provider,
    caption: str = "",
) -> ScreeningResult:
    """Perform a fast screening pass on a figure image.

    Uses the LLM provider's vision capabilities to quickly determine whether
    a figure warrants detailed investigation.

    Args:
        image_path: Path to the figure image file.
        provider: An LLM provider instance with an ``analyze_image`` method.
        caption: Optional figure caption for additional context.

    Returns:
        ScreeningResult indicating whether the figure is suspicious.
    """
    prompt = PROMPT_SCREEN_FIGURE
    if caption:
        prompt += f"\n\nFigure caption: {caption}"

    result = await provider.analyze_image(image_path, prompt, response_schema={"type": "object"})

    # provider.analyze_image returns a dict with 'content', 'parsed', 'model', etc.
    raw_content = result.get("content", "") if isinstance(result, dict) else str(result)
    model_used = result.get("model", "unknown") if isinstance(result, dict) else "unknown"

    parsed = (
        result.get("parsed")
        if isinstance(result, dict) and result.get("parsed")
        else _parse_json_response(raw_content)
    ) or {}

    parse_failed = bool(parsed.get("_parse_failed", False))
    suspicious = bool(parsed.get("suspicious", False))
    reason = str(parsed.get("brief_reason", parsed.get("reason", "")))
    confidence = float(parsed.get("confidence", 0.0))

    return ScreeningResult(
        suspicious=suspicious,
        reason=reason,
        confidence=confidence,
        model_used=str(model_used),
        raw_response=raw_content,
        parse_failed=parse_failed,
    )


async def analyze_figure_detailed(
    image_path: str,
    provider,
    caption: str = "",
    figure_type: str = "",
) -> DetailedAnalysisResult:
    """Perform a detailed analysis of a figure image.

    Uses the LLM provider's vision capabilities to identify specific
    indicators of manipulation or data integrity issues.

    Args:
        image_path: Path to the figure image file.
        provider: An LLM provider instance with an ``analyze_image`` method.
        caption: Optional figure caption for additional context.
        figure_type: Optional classification of the figure type (e.g. "western_blot").

    Returns:
        DetailedAnalysisResult with a list of specific findings.
    """
    prompt = PROMPT_ANALYZE_FIGURE
    if caption:
        prompt += f"\n\nFigure caption: {caption}"
    if figure_type:
        prompt += f"\n\nFigure type: {figure_type}"

    result = await provider.analyze_image(image_path, prompt, response_schema={"type": "object"})

    raw_content = result.get("content", "") if isinstance(result, dict) else str(result)
    model_used = result.get("model", "unknown") if isinstance(result, dict) else "unknown"

    parsed = (
        result.get("parsed")
        if isinstance(result, dict) and result.get("parsed")
        else _parse_json_response(raw_content)
    ) or {}

    parse_failed = bool(parsed.get("_parse_failed", False))
    overall_assessment = str(parsed.get("overall_assessment", ""))
    manipulation_likelihood = float(parsed.get("manipulation_likelihood", 0.0))

    findings: list[VisionFinding] = []
    raw_findings = parsed.get("findings", [])
    if isinstance(raw_findings, list):
        for f in raw_findings:
            if isinstance(f, dict):
                findings.append(
                    VisionFinding(
                        finding_type=str(f.get("type", f.get("finding_type", ""))),
                        description=str(f.get("description", "")),
                        location=str(f.get("location", "")),
                        confidence=float(f.get("confidence", 0.0)),
                    )
                )

    return DetailedAnalysisResult(
        findings=findings,
        overall_assessment=overall_assessment,
        manipulation_likelihood=manipulation_likelihood,
        model_used=str(model_used),
        raw_response=raw_content,
        parse_failed=parse_failed,
    )


async def classify_figure(image_path: str, provider) -> str:
    """Classify the type of a figure image.

    Uses the LLM provider's vision capabilities to determine the figure type
    (e.g. "western_blot", "bar_chart", "microscopy", "flow_cytometry", etc.).

    Args:
        image_path: Path to the figure image file.
        provider: An LLM provider instance with an ``analyze_image`` method.

    Returns:
        A string describing the figure type.
    """
    result = await provider.analyze_image(
        image_path, PROMPT_FIGURE_CLASSIFY, response_schema={"type": "object"}
    )

    raw_content = result.get("content", "") if isinstance(result, dict) else str(result)
    parsed = (
        result.get("parsed")
        if isinstance(result, dict) and result.get("parsed")
        else _parse_json_response(raw_content)
    ) or {}
    figure_type = parsed.get("figure_type", "")

    if not figure_type:
        figure_type = raw_content.strip().strip('"').strip("'")

    return str(figure_type)
