"""Tests for western blot LLM prompt templates."""

from __future__ import annotations

from rosette.llm.prompts_western_blot import (
    PROMPT_WESTERN_BLOT_DETAILED,
    PROMPT_WESTERN_BLOT_SCREEN,
    SYSTEM_WESTERN_BLOT_ANALYST,
)


class TestWesternBlotPrompts:
    def test_system_prompt_is_string(self) -> None:
        assert isinstance(SYSTEM_WESTERN_BLOT_ANALYST, str)
        assert len(SYSTEM_WESTERN_BLOT_ANALYST) > 50

    def test_screen_prompt_mentions_indicators(self) -> None:
        assert "Duplicated bands" in PROMPT_WESTERN_BLOT_SCREEN
        assert "Splice boundaries" in PROMPT_WESTERN_BLOT_SCREEN
        assert "JSON" in PROMPT_WESTERN_BLOT_SCREEN

    def test_detailed_prompt_mentions_analysis(self) -> None:
        assert "Lane-by-lane" in PROMPT_WESTERN_BLOT_DETAILED
        assert "Splice detection" in PROMPT_WESTERN_BLOT_DETAILED
        assert "JSON" in PROMPT_WESTERN_BLOT_DETAILED
