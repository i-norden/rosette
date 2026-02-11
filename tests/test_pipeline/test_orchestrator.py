"""Tests for the pipeline orchestrator with mocked dependencies."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from snoopy.config import SnoopyConfig
from snoopy.db.models import Paper, ProcessingLog
from snoopy.db.session import get_session, init_db
from snoopy.pipeline.orchestrator import PipelineOrchestrator


@pytest.fixture
def orchestrator_config(tmp_path) -> SnoopyConfig:
    """Configuration pointing at an in-memory-like temp SQLite DB."""
    db_path = tmp_path / "orch_test.db"
    return SnoopyConfig(
        storage={
            "database_url": f"sqlite:///{db_path}",
            "pdf_dir": str(tmp_path / "pdfs"),
            "figures_dir": str(tmp_path / "figures"),
            "reports_dir": str(tmp_path / "reports"),
        }
    )


@pytest.fixture
def seeded_db(orchestrator_config) -> str:
    """Initialize DB and seed with a pending paper. Return its paper_id."""
    init_db(orchestrator_config.storage.database_url)

    paper_id = "test-paper-orch-001"
    with get_session() as session:
        paper = Paper(
            id=paper_id,
            title="Test Paper for Orchestrator",
            doi="10.1234/test.orch",
            source="test",
            status="pending",
        )
        session.add(paper)

    return paper_id


class TestProcessPaperFullPipeline:
    @pytest.mark.asyncio
    async def test_process_paper_full_pipeline(
        self, orchestrator_config, seeded_db
    ) -> None:
        """Process a paper through all stages with all external calls mocked.

        Verifies that the orchestrator runs through its stages without error
        when external dependencies (download, LLM, etc.) are mocked out.
        """
        paper_id = seeded_db

        with (
            patch.object(
                PipelineOrchestrator, "_run_download", new_callable=AsyncMock
            ) as mock_download,
            patch.object(
                PipelineOrchestrator, "_run_extract_text", new_callable=AsyncMock
            ) as mock_extract_text,
            patch.object(
                PipelineOrchestrator, "_run_extract_figures", new_callable=AsyncMock
            ) as mock_extract_figures,
            patch.object(
                PipelineOrchestrator, "_run_extract_stats", new_callable=AsyncMock
            ) as mock_extract_stats,
            patch.object(
                PipelineOrchestrator, "_run_classify_figures", new_callable=AsyncMock
            ) as mock_classify,
            patch.object(
                PipelineOrchestrator, "_run_analyze_images", new_callable=AsyncMock
            ) as mock_analyze_images,
            patch.object(
                PipelineOrchestrator, "_run_analyze_stats", new_callable=AsyncMock
            ) as mock_analyze_stats,
            patch.object(
                PipelineOrchestrator, "_run_aggregate", new_callable=AsyncMock
            ) as mock_aggregate,
            patch.object(
                PipelineOrchestrator, "_run_report", new_callable=AsyncMock
            ) as mock_report,
            patch(
                "snoopy.pipeline.orchestrator.ClaudeProvider"
            ) as mock_claude,
        ):
            mock_claude.return_value = MagicMock()

            orchestrator = PipelineOrchestrator(orchestrator_config)
            await orchestrator.process_paper(paper_id)

            # Verify all stages were called
            mock_download.assert_called_once_with(paper_id)
            mock_extract_text.assert_called_once_with(paper_id)
            mock_extract_figures.assert_called_once_with(paper_id)
            mock_extract_stats.assert_called_once_with(paper_id)
            mock_classify.assert_called_once_with(paper_id)
            mock_analyze_images.assert_called_once_with(paper_id)
            mock_analyze_stats.assert_called_once_with(paper_id)
            mock_aggregate.assert_called_once_with(paper_id)
            mock_report.assert_called_once_with(paper_id)

        # Paper should be marked complete
        with get_session() as session:
            paper = session.get(Paper, paper_id)
            assert paper.status == "complete"


class TestStageLogging:
    @pytest.mark.asyncio
    async def test_stage_logging(self, orchestrator_config, seeded_db) -> None:
        """Verify that ProcessingLog entries are created for each stage."""
        paper_id = seeded_db

        with (
            patch.object(
                PipelineOrchestrator, "_run_download", new_callable=AsyncMock
            ),
            patch.object(
                PipelineOrchestrator, "_run_extract_text", new_callable=AsyncMock
            ),
            patch.object(
                PipelineOrchestrator, "_run_extract_figures", new_callable=AsyncMock
            ),
            patch.object(
                PipelineOrchestrator, "_run_extract_stats", new_callable=AsyncMock
            ),
            patch.object(
                PipelineOrchestrator, "_run_classify_figures", new_callable=AsyncMock
            ),
            patch.object(
                PipelineOrchestrator, "_run_analyze_images", new_callable=AsyncMock
            ),
            patch.object(
                PipelineOrchestrator, "_run_analyze_stats", new_callable=AsyncMock
            ),
            patch.object(
                PipelineOrchestrator, "_run_aggregate", new_callable=AsyncMock
            ),
            patch.object(
                PipelineOrchestrator, "_run_report", new_callable=AsyncMock
            ),
            patch(
                "snoopy.pipeline.orchestrator.ClaudeProvider"
            ) as mock_claude,
        ):
            mock_claude.return_value = MagicMock()
            orchestrator = PipelineOrchestrator(orchestrator_config)
            await orchestrator.process_paper(paper_id)

        with get_session() as session:
            from sqlalchemy import select

            logs = session.execute(
                select(ProcessingLog)
                .where(ProcessingLog.paper_id == paper_id)
                .order_by(ProcessingLog.id)
            ).scalars().all()

            # Each stage should have a "started" and "completed" entry
            stages_logged = set()
            for log in logs:
                stages_logged.add(log.stage)
                assert log.status in ("started", "completed")

            # All pipeline stages (minus discover/prioritize) should be logged
            expected_stages = {
                "download", "extract_text", "extract_figures", "extract_stats",
                "classify_figures", "analyze_images", "analyze_stats",
                "aggregate", "report",
            }
            assert expected_stages == stages_logged


class TestBuildMethodWeights:
    def test_build_method_weights(self, orchestrator_config) -> None:
        """Verify the weights dict is built correctly from config."""
        with patch(
            "snoopy.pipeline.orchestrator.ClaudeProvider"
        ) as mock_claude:
            mock_claude.return_value = MagicMock()
            orchestrator = PipelineOrchestrator(orchestrator_config)

        weights = orchestrator._build_method_weights()

        assert "clone_detection" in weights
        assert "phash" in weights
        assert "pvalue_check" in weights
        assert "grim" in weights
        assert "noise_analysis" in weights
        assert "ela" in weights
        assert "benford" in weights
        assert "duplicate_values" in weights
        assert "llm_vision" in weights

        # Verify weights match config values
        cfg = orchestrator_config.analysis
        assert weights["clone_detection"] == cfg.weight_clone_detection
        assert weights["ela"] == cfg.weight_ela
        assert weights["llm_vision"] == cfg.weight_llm_vision
