"""Tests for LLM vision analysis with mocked provider."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from snoopy.analysis.llm_vision import (
    DetailedAnalysisResult,
    ScreeningResult,
    _parse_json_response,
    analyze_figure_detailed,
    classify_figure,
    screen_figure,
)


def _make_mock_provider(response: dict) -> AsyncMock:
    """Create a mock LLM provider returning the given response dict."""
    provider = AsyncMock()
    provider.analyze_image = AsyncMock(return_value=response)
    return provider


class TestScreenFigure:
    @pytest.mark.asyncio
    async def test_screen_figure_suspicious(self, sample_image: str) -> None:
        """Screening returns suspicious=True when the mock LLM says so."""
        provider = _make_mock_provider(
            {
                "content": json.dumps(
                    {
                        "suspicious": True,
                        "brief_reason": "Duplicated band pattern detected",
                        "confidence": 0.85,
                    }
                ),
                "parsed": {
                    "suspicious": True,
                    "brief_reason": "Duplicated band pattern detected",
                    "confidence": 0.85,
                },
                "model": "test-model",
            }
        )

        result = await screen_figure(sample_image, provider)
        assert isinstance(result, ScreeningResult)
        assert result.suspicious is True
        assert result.confidence == 0.85
        assert "Duplicated" in result.reason
        assert result.model_used == "test-model"

    @pytest.mark.asyncio
    async def test_screen_figure_clean(self, sample_image: str) -> None:
        """Screening returns suspicious=False when the mock LLM says clean."""
        provider = _make_mock_provider(
            {
                "content": json.dumps(
                    {
                        "suspicious": False,
                        "brief_reason": "No anomalies detected",
                        "confidence": 0.1,
                    }
                ),
                "parsed": {
                    "suspicious": False,
                    "brief_reason": "No anomalies detected",
                    "confidence": 0.1,
                },
                "model": "test-model",
            }
        )

        result = await screen_figure(sample_image, provider)
        assert isinstance(result, ScreeningResult)
        assert result.suspicious is False
        assert result.confidence == 0.1


class TestAnalyzeFigureDetailed:
    @pytest.mark.asyncio
    async def test_analyze_figure_detailed_with_findings(self, sample_image: str) -> None:
        """Detailed analysis returns findings from mock LLM response."""
        provider = _make_mock_provider(
            {
                "content": json.dumps(
                    {
                        "findings": [
                            {
                                "type": "duplication",
                                "description": "Lane 3 appears identical to lane 5",
                                "location": "center of image",
                                "confidence": 0.9,
                            },
                            {
                                "type": "splice",
                                "description": "Abrupt background change between panels",
                                "location": "panel boundary",
                                "confidence": 0.7,
                            },
                        ],
                        "overall_assessment": "Likely manipulated",
                        "manipulation_likelihood": 0.85,
                    }
                ),
                "parsed": {
                    "findings": [
                        {
                            "type": "duplication",
                            "description": "Lane 3 appears identical to lane 5",
                            "location": "center of image",
                            "confidence": 0.9,
                        },
                        {
                            "type": "splice",
                            "description": "Abrupt background change between panels",
                            "location": "panel boundary",
                            "confidence": 0.7,
                        },
                    ],
                    "overall_assessment": "Likely manipulated",
                    "manipulation_likelihood": 0.85,
                },
                "model": "test-model",
            }
        )

        result = await analyze_figure_detailed(
            sample_image, provider, caption="Figure 1", figure_type="western_blot"
        )
        assert isinstance(result, DetailedAnalysisResult)
        assert len(result.findings) == 2
        assert result.findings[0].finding_type == "duplication"
        assert result.findings[0].confidence == 0.9
        assert result.findings[1].finding_type == "splice"
        assert result.manipulation_likelihood == 0.85
        assert result.overall_assessment == "Likely manipulated"


class TestParseJsonResponse:
    def test_parse_json_response_from_code_fence(self) -> None:
        """Parse JSON wrapped in markdown code fences."""
        raw = '```json\n{"suspicious": true, "confidence": 0.9}\n```'
        parsed = _parse_json_response(raw)
        assert parsed["suspicious"] is True
        assert parsed["confidence"] == 0.9

    def test_parse_json_response_direct(self) -> None:
        """Parse raw JSON without code fences."""
        raw = '{"figure_type": "western_blot", "confidence": 0.95}'
        parsed = _parse_json_response(raw)
        assert parsed["figure_type"] == "western_blot"

    def test_parse_json_response_with_extra_text(self) -> None:
        """Parse JSON embedded in extra surrounding text."""
        raw = 'Here is my analysis:\n{"result": true}\nEnd of analysis.'
        parsed = _parse_json_response(raw)
        assert parsed["result"] is True

    def test_parse_json_response_invalid(self) -> None:
        """Return dict with _parse_failed flag when JSON cannot be parsed."""
        raw = "This is not JSON at all, no braces here."
        parsed = _parse_json_response(raw)
        assert parsed == {"_parse_failed": True}


class TestClassifyFigure:
    @pytest.mark.asyncio
    async def test_classify_figure(self, sample_image: str) -> None:
        """Classify figure type from mock LLM response."""
        provider = _make_mock_provider(
            {
                "content": json.dumps(
                    {
                        "figure_type": "western_blot",
                        "confidence": 0.92,
                    }
                ),
                "parsed": {
                    "figure_type": "western_blot",
                    "confidence": 0.92,
                },
                "model": "test-model",
            }
        )

        result = await classify_figure(sample_image, provider)
        assert result == "western_blot"
