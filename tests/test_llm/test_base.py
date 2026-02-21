"""Tests for LLM base module."""

from __future__ import annotations

from rosette.llm.base import LLMProvider, LLMResponse


class TestLLMResponse:
    def test_llm_response_creation(self) -> None:
        """LLMResponse dataclass can be instantiated."""
        resp = LLMResponse(
            content="test content",
            parsed={"key": "value"},
            model="claude-3",
            input_tokens=100,
            output_tokens=50,
        )
        assert resp.content == "test content"
        assert resp.parsed == {"key": "value"}
        assert resp.model == "claude-3"
        assert resp.input_tokens == 100
        assert resp.output_tokens == 50

    def test_llm_response_with_none_parsed(self) -> None:
        """LLMResponse with None parsed."""
        resp = LLMResponse(
            content="raw text",
            parsed=None,
            model="model",
            input_tokens=10,
            output_tokens=20,
        )
        assert resp.parsed is None


class TestLLMProvider:
    def test_llm_provider_is_protocol(self) -> None:
        """LLMProvider is a Protocol class."""
        assert hasattr(LLMProvider, "analyze_image")
        assert hasattr(LLMProvider, "analyze_text")
        assert hasattr(LLMProvider, "analyze_images_batch")
