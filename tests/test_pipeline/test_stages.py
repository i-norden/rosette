"""Tests for rosette.pipeline.stages module."""

from __future__ import annotations

from rosette.pipeline.stages import PIPELINE_STAGES, get_next_stage, get_remaining_stages


class TestGetNextStage:
    def test_none_returns_first(self):
        assert get_next_stage(None) == PIPELINE_STAGES[0]

    def test_first_stage_returns_second(self):
        assert get_next_stage(PIPELINE_STAGES[0]) == PIPELINE_STAGES[1]

    def test_last_stage_returns_none(self):
        assert get_next_stage(PIPELINE_STAGES[-1]) is None

    def test_middle_stage(self):
        mid = len(PIPELINE_STAGES) // 2
        assert get_next_stage(PIPELINE_STAGES[mid]) == PIPELINE_STAGES[mid + 1]

    def test_unknown_stage_returns_first(self):
        assert get_next_stage("nonexistent_stage") == PIPELINE_STAGES[0]


class TestGetRemainingStages:
    def test_none_returns_all(self):
        remaining = get_remaining_stages(None)
        assert remaining == PIPELINE_STAGES

    def test_first_stage_returns_rest(self):
        remaining = get_remaining_stages(PIPELINE_STAGES[0])
        assert remaining == PIPELINE_STAGES[1:]

    def test_last_stage_returns_empty(self):
        remaining = get_remaining_stages(PIPELINE_STAGES[-1])
        assert remaining == []

    def test_penultimate_stage(self):
        remaining = get_remaining_stages(PIPELINE_STAGES[-2])
        assert remaining == [PIPELINE_STAGES[-1]]

    def test_unknown_stage_returns_all(self):
        remaining = get_remaining_stages("nonexistent_stage")
        assert remaining == PIPELINE_STAGES

    def test_returns_list_copy(self):
        """Ensure modifications to the returned list don't affect the original."""
        remaining = get_remaining_stages(None)
        remaining.clear()
        assert len(PIPELINE_STAGES) > 0


class TestPipelineStagesOrder:
    def test_stages_are_ordered(self):
        """Verify expected stage ordering."""
        assert PIPELINE_STAGES.index("download") < PIPELINE_STAGES.index("extract_text")
        assert PIPELINE_STAGES.index("extract_text") < PIPELINE_STAGES.index("extract_figures")
        assert PIPELINE_STAGES.index("analyze_images_auto") < PIPELINE_STAGES.index("aggregate")
        assert PIPELINE_STAGES.index("analyze_images_llm") < PIPELINE_STAGES.index("aggregate")
        assert PIPELINE_STAGES.index("aggregate") < PIPELINE_STAGES.index("report")

    def test_discover_is_first(self):
        assert PIPELINE_STAGES[0] == "discover"

    def test_report_is_last(self):
        assert PIPELINE_STAGES[-1] == "report"
