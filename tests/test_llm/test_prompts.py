"""Tests for LLM prompt templates."""

from rosette.llm.prompts import (
    PROMPT_ANALYZE_FIGURE,
    PROMPT_FIGURE_CLASSIFY,
    PROMPT_SCREEN_FIGURE,
    PROMPT_SUMMARIZE_EVIDENCE,
    SYSTEM_FORENSIC_ANALYST,
    SYSTEM_IMAGE_INTEGRITY,
    SYSTEM_PROOF_WRITER,
    SYSTEM_STATISTICAL_ANALYST,
)


class TestPromptsExist:
    def test_system_prompts_are_strings(self):
        assert isinstance(SYSTEM_IMAGE_INTEGRITY, str)
        assert isinstance(SYSTEM_FORENSIC_ANALYST, str)
        assert isinstance(SYSTEM_STATISTICAL_ANALYST, str)
        assert isinstance(SYSTEM_PROOF_WRITER, str)

    def test_prompts_are_non_empty(self):
        assert len(PROMPT_FIGURE_CLASSIFY) > 50
        assert len(PROMPT_SCREEN_FIGURE) > 50
        assert len(PROMPT_ANALYZE_FIGURE) > 50
        assert len(PROMPT_SUMMARIZE_EVIDENCE) > 50

    def test_prompts_mention_json(self):
        assert "JSON" in PROMPT_FIGURE_CLASSIFY or "json" in PROMPT_FIGURE_CLASSIFY
        assert "JSON" in PROMPT_SCREEN_FIGURE or "json" in PROMPT_SCREEN_FIGURE

    def test_summarize_has_placeholder(self):
        assert "{findings_json}" in PROMPT_SUMMARIZE_EVIDENCE

    def test_summarize_format_works(self):
        result = PROMPT_SUMMARIZE_EVIDENCE.format(findings_json='[{"test": true}]')
        assert '[{"test": true}]' in result
