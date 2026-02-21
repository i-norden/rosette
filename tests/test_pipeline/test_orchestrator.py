"""Tests for the pipeline orchestrator with mocked dependencies."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rosette.config import RosetteConfig
from rosette.db.models import Paper, ProcessingLog
from rosette.db.session import get_session, init_async_db, init_db
from rosette.pipeline.orchestrator import PipelineOrchestrator


@pytest.fixture
def orchestrator_config(tmp_path) -> RosetteConfig:
    """Configuration pointing at an in-memory-like temp SQLite DB."""
    db_path = tmp_path / "orch_test.db"
    return RosetteConfig(
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
    init_async_db(orchestrator_config.storage.database_url)

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


def _mock_all_stages():
    """Return a context manager that mocks all orchestrator stage handlers."""
    return (
        patch.object(PipelineOrchestrator, "_run_download", new_callable=AsyncMock),
        patch.object(PipelineOrchestrator, "_run_extract_text", new_callable=AsyncMock),
        patch.object(PipelineOrchestrator, "_run_extract_figures", new_callable=AsyncMock),
        patch.object(PipelineOrchestrator, "_run_extract_stats", new_callable=AsyncMock),
        patch.object(PipelineOrchestrator, "_run_classify_figures", new_callable=AsyncMock),
        patch.object(PipelineOrchestrator, "_run_analyze_images_auto", new_callable=AsyncMock),
        patch.object(PipelineOrchestrator, "_run_analyze_images_llm", new_callable=AsyncMock),
        patch.object(PipelineOrchestrator, "_run_analyze_stats", new_callable=AsyncMock),
        patch.object(PipelineOrchestrator, "_run_aggregate", new_callable=AsyncMock),
        patch.object(PipelineOrchestrator, "_run_report", new_callable=AsyncMock),
        patch("rosette.pipeline.orchestrator.ClaudeProvider"),
    )


class TestProcessPaperFullPipeline:
    @pytest.mark.asyncio
    async def test_process_paper_full_pipeline(self, orchestrator_config, seeded_db) -> None:
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
                PipelineOrchestrator, "_run_analyze_images_auto", new_callable=AsyncMock
            ) as mock_analyze_images_auto,
            patch.object(
                PipelineOrchestrator, "_run_analyze_images_llm", new_callable=AsyncMock
            ) as mock_analyze_images_llm,
            patch.object(
                PipelineOrchestrator, "_run_analyze_stats", new_callable=AsyncMock
            ) as mock_analyze_stats,
            patch.object(
                PipelineOrchestrator, "_run_aggregate", new_callable=AsyncMock
            ) as mock_aggregate,
            patch.object(
                PipelineOrchestrator, "_run_report", new_callable=AsyncMock
            ) as mock_report,
            patch("rosette.pipeline.orchestrator.ClaudeProvider") as mock_claude,
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
            mock_analyze_images_auto.assert_called_once_with(paper_id)
            mock_analyze_images_llm.assert_called_once_with(paper_id)
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
            patch.object(PipelineOrchestrator, "_run_download", new_callable=AsyncMock),
            patch.object(PipelineOrchestrator, "_run_extract_text", new_callable=AsyncMock),
            patch.object(PipelineOrchestrator, "_run_extract_figures", new_callable=AsyncMock),
            patch.object(PipelineOrchestrator, "_run_extract_stats", new_callable=AsyncMock),
            patch.object(PipelineOrchestrator, "_run_classify_figures", new_callable=AsyncMock),
            patch.object(PipelineOrchestrator, "_run_analyze_images_auto", new_callable=AsyncMock),
            patch.object(PipelineOrchestrator, "_run_analyze_images_llm", new_callable=AsyncMock),
            patch.object(PipelineOrchestrator, "_run_analyze_stats", new_callable=AsyncMock),
            patch.object(PipelineOrchestrator, "_run_aggregate", new_callable=AsyncMock),
            patch.object(PipelineOrchestrator, "_run_report", new_callable=AsyncMock),
            patch("rosette.pipeline.orchestrator.ClaudeProvider") as mock_claude,
        ):
            mock_claude.return_value = MagicMock()
            orchestrator = PipelineOrchestrator(orchestrator_config)
            await orchestrator.process_paper(paper_id)

        with get_session() as session:
            from sqlalchemy import select

            logs = (
                session.execute(
                    select(ProcessingLog)
                    .where(ProcessingLog.paper_id == paper_id)
                    .order_by(ProcessingLog.id)
                )
                .scalars()
                .all()
            )

            # Each stage should have a "started" and "completed" entry
            stages_logged = set()
            for log in logs:
                stages_logged.add(log.stage)
                assert log.status in ("started", "completed")

            # All pipeline stages (minus discover/prioritize) should be logged
            expected_stages = {
                "download",
                "extract_text",
                "extract_figures",
                "extract_stats",
                "classify_figures",
                "analyze_images_auto",
                "analyze_images_llm",
                "analyze_stats",
                "aggregate",
                "report",
            }
            assert expected_stages == stages_logged


class TestBuildMethodWeights:
    def test_build_method_weights(self, orchestrator_config) -> None:
        """Verify the weights dict is built correctly from config."""
        with patch("rosette.pipeline.orchestrator.ClaudeProvider") as mock_claude:
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


class TestStageFailureAndResumability:
    @pytest.mark.asyncio
    async def test_stage_failure_marks_paper_as_error(self, orchestrator_config, seeded_db) -> None:
        """When a stage fails, the paper should be marked as 'error'."""
        paper_id = seeded_db

        with (
            patch.object(PipelineOrchestrator, "_run_download", new_callable=AsyncMock),
            patch.object(
                PipelineOrchestrator,
                "_run_extract_text",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Text extraction failed"),
            ),
            patch("rosette.pipeline.orchestrator.ClaudeProvider") as mock_claude,
        ):
            mock_claude.return_value = MagicMock()
            orchestrator = PipelineOrchestrator(orchestrator_config)

            with pytest.raises(RuntimeError, match="Text extraction failed"):
                await orchestrator.process_paper(paper_id)

        with get_session() as session:
            paper = session.get(Paper, paper_id)
            assert paper.status == "error"
            assert "Text extraction failed" in (paper.error_message or "")

    @pytest.mark.asyncio
    async def test_resume_from_last_completed_stage(self, orchestrator_config, seeded_db) -> None:
        """After a failure, resuming should start from the last completed stage."""
        paper_id = seeded_db

        # First run: succeeds through download, fails at extract_text
        with (
            patch.object(PipelineOrchestrator, "_run_download", new_callable=AsyncMock),
            patch.object(
                PipelineOrchestrator,
                "_run_extract_text",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            patch("rosette.pipeline.orchestrator.ClaudeProvider") as mock_claude,
        ):
            mock_claude.return_value = MagicMock()
            orchestrator = PipelineOrchestrator(orchestrator_config)

            with pytest.raises(RuntimeError):
                await orchestrator.process_paper(paper_id)

        # Second run: all stages succeed - should resume from extract_text
        with (
            patch.object(
                PipelineOrchestrator, "_run_download", new_callable=AsyncMock
            ) as mock_download,
            patch.object(
                PipelineOrchestrator, "_run_extract_text", new_callable=AsyncMock
            ) as mock_extract_text,
            patch.object(PipelineOrchestrator, "_run_extract_figures", new_callable=AsyncMock),
            patch.object(PipelineOrchestrator, "_run_extract_stats", new_callable=AsyncMock),
            patch.object(PipelineOrchestrator, "_run_classify_figures", new_callable=AsyncMock),
            patch.object(PipelineOrchestrator, "_run_analyze_images_auto", new_callable=AsyncMock),
            patch.object(PipelineOrchestrator, "_run_analyze_images_llm", new_callable=AsyncMock),
            patch.object(PipelineOrchestrator, "_run_analyze_stats", new_callable=AsyncMock),
            patch.object(PipelineOrchestrator, "_run_aggregate", new_callable=AsyncMock),
            patch.object(PipelineOrchestrator, "_run_report", new_callable=AsyncMock),
            patch("rosette.pipeline.orchestrator.ClaudeProvider") as mock_claude,
        ):
            mock_claude.return_value = MagicMock()
            orchestrator = PipelineOrchestrator(orchestrator_config)
            await orchestrator.process_paper(paper_id)

            # Download was already completed, so it should NOT be called again
            mock_download.assert_not_called()
            # extract_text should be called (it was the failed stage)
            mock_extract_text.assert_called_once_with(paper_id)

        with get_session() as session:
            paper = session.get(Paper, paper_id)
            assert paper.status == "complete"

    @pytest.mark.asyncio
    async def test_stage_failure_logs_failed_status(self, orchestrator_config, seeded_db) -> None:
        """A failed stage should have a 'failed' ProcessingLog entry."""
        paper_id = seeded_db

        with (
            patch.object(PipelineOrchestrator, "_run_download", new_callable=AsyncMock),
            patch.object(
                PipelineOrchestrator,
                "_run_extract_text",
                new_callable=AsyncMock,
                side_effect=RuntimeError("extraction error"),
            ),
            patch("rosette.pipeline.orchestrator.ClaudeProvider") as mock_claude,
        ):
            mock_claude.return_value = MagicMock()
            orchestrator = PipelineOrchestrator(orchestrator_config)

            with pytest.raises(RuntimeError):
                await orchestrator.process_paper(paper_id)

        with get_session() as session:
            from sqlalchemy import select

            logs = (
                session.execute(
                    select(ProcessingLog)
                    .where(ProcessingLog.paper_id == paper_id)
                    .where(ProcessingLog.status == "failed")
                )
                .scalars()
                .all()
            )
            assert len(logs) == 1
            assert logs[0].stage == "extract_text"
            assert "extraction error" in (logs[0].details or "")


class TestRunBatch:
    @pytest.mark.asyncio
    async def test_run_batch_processes_pending_papers(self, orchestrator_config, seeded_db):
        """run_batch should pick up pending papers with pdf_path set."""
        paper_id = seeded_db
        # Set pdf_path so the query filter matches
        with get_session() as session:
            paper = session.get(Paper, paper_id)
            paper.pdf_path = "/tmp/fake.pdf"
            paper.priority_score = 50.0

        with (
            patch.object(
                PipelineOrchestrator, "process_paper", new_callable=AsyncMock
            ) as mock_process,
            patch("rosette.pipeline.orchestrator.ClaudeProvider") as mock_claude,
        ):
            mock_claude.return_value = MagicMock()
            orchestrator = PipelineOrchestrator(orchestrator_config)
            results = await orchestrator.run_batch(limit=10)

        assert paper_id in results
        mock_process.assert_called_once_with(paper_id)

    @pytest.mark.asyncio
    async def test_run_batch_min_priority_filters(self, orchestrator_config, seeded_db):
        """run_batch with min_priority should skip low-priority papers."""
        paper_id = seeded_db
        with get_session() as session:
            paper = session.get(Paper, paper_id)
            paper.pdf_path = "/tmp/fake.pdf"
            paper.priority_score = 10.0  # Below threshold

        with patch("rosette.pipeline.orchestrator.ClaudeProvider") as mock_claude:
            mock_claude.return_value = MagicMock()
            orchestrator = PipelineOrchestrator(orchestrator_config)
            results = await orchestrator.run_batch(limit=10, min_priority=50.0)

        assert results == []

    @pytest.mark.asyncio
    async def test_run_batch_sqlite_forces_serial(self, orchestrator_config, seeded_db):
        """SQLite DB URL should force concurrency to 1."""
        paper_id = seeded_db
        with get_session() as session:
            paper = session.get(Paper, paper_id)
            paper.pdf_path = "/tmp/fake.pdf"
            paper.priority_score = 50.0

        with (
            patch.object(
                PipelineOrchestrator, "process_paper", new_callable=AsyncMock
            ) as mock_process,
            patch("rosette.pipeline.orchestrator.ClaudeProvider") as mock_claude,
        ):
            mock_claude.return_value = MagicMock()
            orchestrator = PipelineOrchestrator(orchestrator_config)
            # orchestrator_config uses sqlite, so concurrency should be forced to 1
            await orchestrator.run_batch(limit=10)

        mock_process.assert_called_once_with(paper_id)
