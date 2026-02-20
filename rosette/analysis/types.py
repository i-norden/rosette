"""Shared type definitions for analysis findings."""

from __future__ import annotations

from typing import Any, TypedDict


class FindingDict(TypedDict, total=False):
    """Standardized finding dictionary returned by all analysis methods.

    Required keys: title, analysis_type, method, severity, confidence.
    Optional keys: description, figure_id, evidence.
    """

    title: str
    analysis_type: str
    method: str
    severity: str
    confidence: float
    description: str
    figure_id: str
    evidence: dict[str, Any]
