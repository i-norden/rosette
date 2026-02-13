"""Tests for snoopy.campaign.orchestrator module.

Tests the top-level campaign coordinator: mode dispatching,
state transitions, pause/resume, and batch processing.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from snoopy.campaign.orchestrator import CampaignOrchestrator
from snoopy.config import SnoopyConfig
from snoopy.db.models import Campaign, CampaignPaper, Paper
from snoopy.db.session import get_session, init_async_db, init_db


@pytest.fixture
def orch_config(tmp_path) -> SnoopyConfig:
    db_path = tmp_path / "campaign_orch_test.db"
    return SnoopyConfig(
        storage={
            "database_url": f"sqlite:///{db_path}",
            "pdf_dir": str(tmp_path / "pdfs"),
            "figures_dir": str(tmp_path / "figures"),
            "reports_dir": str(tmp_path / "reports"),
        },
        campaign={
            "auto_risk_promotion_threshold": 30.0,
            "batch_concurrency": 2,
        },
    )


def _seed_campaign(
    config: SnoopyConfig,
    mode: str,
    campaign_id: str = "test-campaign-001",
    status: str = "created",
    seed_dois: list[str] | None = None,
    max_depth: int = 2,
    max_papers: int = 100,
    llm_budget: int = 10,
) -> str:
    """Create a campaign in the DB. Returns campaign_id."""
    init_db(config.storage.database_url)
    init_async_db(config.storage.database_url)

    with get_session() as session:
        campaign = Campaign(
            id=campaign_id,
            name="Test Campaign",
            mode=mode,
            status=status,
            seed_dois=json.dumps(seed_dois) if seed_dois else None,
            max_depth=max_depth,
            max_papers=max_papers,
            llm_budget=llm_budget,
            config_json=json.dumps({"field": "biomedical", "min_citations": 50}),
        )
        session.add(campaign)

    return campaign_id


class TestCampaignModeDispatching:
    @pytest.mark.asyncio
    async def test_network_expansion_mode(self, orch_config):
        campaign_id = _seed_campaign(orch_config, "network_expansion", seed_dois=["10.1234/seed"])

        with (
            patch.object(
                CampaignOrchestrator, "_run_network_expansion", new_callable=AsyncMock
            ) as mock_ne,
            patch.object(
                CampaignOrchestrator, "_run_domain_scan", new_callable=AsyncMock
            ) as mock_ds,
            patch.object(
                CampaignOrchestrator, "_run_paper_mill", new_callable=AsyncMock
            ) as mock_pm,
        ):
            orchestrator = CampaignOrchestrator(orch_config, campaign_id)
            await orchestrator.run()

            mock_ne.assert_called_once()
            mock_ds.assert_not_called()
            mock_pm.assert_not_called()

    @pytest.mark.asyncio
    async def test_domain_scan_mode(self, orch_config):
        campaign_id = _seed_campaign(orch_config, "domain_scan")

        with (
            patch.object(
                CampaignOrchestrator, "_run_network_expansion", new_callable=AsyncMock
            ) as mock_ne,
            patch.object(
                CampaignOrchestrator, "_run_domain_scan", new_callable=AsyncMock
            ) as mock_ds,
            patch.object(
                CampaignOrchestrator, "_run_paper_mill", new_callable=AsyncMock
            ) as mock_pm,
        ):
            orchestrator = CampaignOrchestrator(orch_config, campaign_id)
            await orchestrator.run()

            mock_ne.assert_not_called()
            mock_ds.assert_called_once()
            mock_pm.assert_not_called()

    @pytest.mark.asyncio
    async def test_paper_mill_mode(self, orch_config):
        campaign_id = _seed_campaign(orch_config, "paper_mill", seed_dois=["10.1234/seed"])

        with (
            patch.object(
                CampaignOrchestrator, "_run_network_expansion", new_callable=AsyncMock
            ) as mock_ne,
            patch.object(
                CampaignOrchestrator, "_run_domain_scan", new_callable=AsyncMock
            ) as mock_ds,
            patch.object(
                CampaignOrchestrator, "_run_paper_mill", new_callable=AsyncMock
            ) as mock_pm,
        ):
            orchestrator = CampaignOrchestrator(orch_config, campaign_id)
            await orchestrator.run()

            mock_ne.assert_not_called()
            mock_ds.assert_not_called()
            mock_pm.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_mode_raises(self, orch_config):
        campaign_id = _seed_campaign(orch_config, "unknown_mode")

        orchestrator = CampaignOrchestrator(orch_config, campaign_id)
        with pytest.raises(ValueError, match="Unknown campaign mode"):
            await orchestrator.run()


class TestCampaignStatusTransitions:
    @pytest.mark.asyncio
    async def test_completed_campaign_skips_run(self, orch_config):
        campaign_id = _seed_campaign(orch_config, "domain_scan", status="completed")

        with patch.object(
            CampaignOrchestrator, "_run_domain_scan", new_callable=AsyncMock
        ) as mock_ds:
            orchestrator = CampaignOrchestrator(orch_config, campaign_id)
            await orchestrator.run()
            mock_ds.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_sets_completed_status(self, orch_config):
        campaign_id = _seed_campaign(orch_config, "domain_scan")

        with patch.object(CampaignOrchestrator, "_run_domain_scan", new_callable=AsyncMock):
            orchestrator = CampaignOrchestrator(orch_config, campaign_id)
            await orchestrator.run()

        with get_session() as session:
            campaign = session.get(Campaign, campaign_id)
            assert campaign.status == "completed"

    @pytest.mark.asyncio
    async def test_nonexistent_campaign_raises(self, orch_config):
        init_db(orch_config.storage.database_url)
        init_async_db(orch_config.storage.database_url)

        orchestrator = CampaignOrchestrator(orch_config, "nonexistent-id")
        with pytest.raises(ValueError, match="not found"):
            await orchestrator.run()


class TestPauseResume:
    @pytest.mark.asyncio
    async def test_pause_sets_status(self, orch_config):
        campaign_id = _seed_campaign(orch_config, "domain_scan", status="auto_analyzing")

        orchestrator = CampaignOrchestrator(orch_config, campaign_id)
        await orchestrator.pause()

        with get_session() as session:
            campaign = session.get(Campaign, campaign_id)
            assert campaign.status == "paused"

    @pytest.mark.asyncio
    async def test_paused_campaign_does_not_complete(self, orch_config):
        campaign_id = _seed_campaign(orch_config, "domain_scan")

        async def _mock_domain_scan(self_inner):
            self_inner._paused = True

        with patch.object(CampaignOrchestrator, "_run_domain_scan", _mock_domain_scan):
            orchestrator = CampaignOrchestrator(orch_config, campaign_id)
            await orchestrator.run()

        with get_session() as session:
            campaign = session.get(Campaign, campaign_id)
            # Should NOT be "completed" since it was paused
            assert campaign.status != "completed"


class TestSeedPapers:
    @pytest.mark.asyncio
    async def test_seed_papers_creates_records(self, orch_config):
        campaign_id = _seed_campaign(
            orch_config,
            "network_expansion",
            seed_dois=["10.1234/seed1", "10.1234/seed2"],
        )

        orchestrator = CampaignOrchestrator(orch_config, campaign_id)
        await orchestrator._seed_papers()

        with get_session() as session:
            from sqlalchemy import select

            cps = (
                session.execute(
                    select(CampaignPaper).where(CampaignPaper.campaign_id == campaign_id)
                )
                .scalars()
                .all()
            )
            assert len(cps) == 2
            for cp in cps:
                assert cp.source == "seed"
                assert cp.depth == 0
                assert cp.triage_status == "pending"

    @pytest.mark.asyncio
    async def test_seed_papers_idempotent(self, orch_config):
        campaign_id = _seed_campaign(
            orch_config,
            "network_expansion",
            seed_dois=["10.1234/seed1"],
        )

        orchestrator = CampaignOrchestrator(orch_config, campaign_id)
        await orchestrator._seed_papers()
        await orchestrator._seed_papers()  # Second call should not duplicate

        with get_session() as session:
            from sqlalchemy import select

            cps = (
                session.execute(
                    select(CampaignPaper).where(CampaignPaper.campaign_id == campaign_id)
                )
                .scalars()
                .all()
            )
            assert len(cps) == 1

    @pytest.mark.asyncio
    async def test_seed_papers_no_seeds(self, orch_config):
        campaign_id = _seed_campaign(orch_config, "domain_scan")

        orchestrator = CampaignOrchestrator(orch_config, campaign_id)
        await orchestrator._seed_papers()

        with get_session() as session:
            from sqlalchemy import select

            cps = (
                session.execute(
                    select(CampaignPaper).where(CampaignPaper.campaign_id == campaign_id)
                )
                .scalars()
                .all()
            )
            assert len(cps) == 0

    @pytest.mark.asyncio
    async def test_seed_papers_reuses_existing_paper_by_doi(self, orch_config):
        campaign_id = _seed_campaign(
            orch_config,
            "network_expansion",
            seed_dois=["10.1234/existing"],
        )

        # Pre-create a paper with the same DOI
        existing_paper_id = "existing-paper-001"
        with get_session() as session:
            session.add(
                Paper(
                    id=existing_paper_id,
                    doi="10.1234/existing",
                    title="Existing Paper",
                    source="manual",
                    status="complete",
                )
            )

        orchestrator = CampaignOrchestrator(orch_config, campaign_id)
        await orchestrator._seed_papers()

        with get_session() as session:
            from sqlalchemy import select

            cp = (
                session.execute(
                    select(CampaignPaper).where(CampaignPaper.campaign_id == campaign_id)
                )
                .scalars()
                .first()
            )
            assert str(cp.paper_id) == existing_paper_id


class TestProcessBatchAuto:
    @pytest.mark.asyncio
    async def test_batch_auto_returns_scores(self, orch_config):
        campaign_id = _seed_campaign(orch_config, "domain_scan")

        # Create papers
        with get_session() as session:
            for i in range(3):
                pid = f"batch-paper-{i}"
                session.add(
                    Paper(
                        id=pid,
                        title=f"Paper {i}",
                        source="test",
                        status="pending",
                    )
                )
                session.add(
                    CampaignPaper(
                        campaign_id=campaign_id,
                        paper_id=pid,
                        source="domain_scan",
                        depth=0,
                        triage_status="pending",
                    )
                )

        paper_ids = [f"batch-paper-{i}" for i in range(3)]

        with patch("snoopy.campaign.orchestrator.TriagePipeline") as MockTriage:
            mock_triage = MockTriage.return_value
            mock_triage.run_auto_tier = AsyncMock(side_effect=[10.0, 40.0, 80.0])

            orchestrator = CampaignOrchestrator(orch_config, campaign_id)
            orchestrator.triage = mock_triage
            results = await orchestrator._process_batch_auto(paper_ids)

        assert len(results) == 3
        assert results["batch-paper-0"] == 10.0
        assert results["batch-paper-1"] == 40.0
        assert results["batch-paper-2"] == 80.0

    @pytest.mark.asyncio
    async def test_batch_auto_updates_flagged_count(self, orch_config):
        campaign_id = _seed_campaign(orch_config, "domain_scan")

        with get_session() as session:
            session.add(
                Paper(
                    id="flag-paper",
                    title="Flaggy",
                    source="test",
                    status="pending",
                )
            )
            session.add(
                CampaignPaper(
                    campaign_id=campaign_id,
                    paper_id="flag-paper",
                    source="domain_scan",
                    depth=0,
                    triage_status="pending",
                )
            )

        with patch("snoopy.campaign.orchestrator.TriagePipeline") as MockTriage:
            mock_triage = MockTriage.return_value
            mock_triage.run_auto_tier = AsyncMock(return_value=50.0)

            orchestrator = CampaignOrchestrator(orch_config, campaign_id)
            orchestrator.triage = mock_triage
            await orchestrator._process_batch_auto(["flag-paper"])

        with get_session() as session:
            campaign = session.get(Campaign, campaign_id)
            assert (campaign.papers_flagged or 0) >= 1

    @pytest.mark.asyncio
    async def test_batch_auto_handles_errors(self, orch_config):
        campaign_id = _seed_campaign(orch_config, "domain_scan")

        with get_session() as session:
            session.add(
                Paper(
                    id="error-paper",
                    title="Error Paper",
                    source="test",
                    status="pending",
                )
            )
            session.add(
                CampaignPaper(
                    campaign_id=campaign_id,
                    paper_id="error-paper",
                    source="domain_scan",
                    depth=0,
                    triage_status="pending",
                )
            )

        with patch("snoopy.campaign.orchestrator.TriagePipeline") as MockTriage:
            mock_triage = MockTriage.return_value
            mock_triage.run_auto_tier = AsyncMock(side_effect=RuntimeError("fail"))

            orchestrator = CampaignOrchestrator(orch_config, campaign_id)
            orchestrator.triage = mock_triage
            results = await orchestrator._process_batch_auto(["error-paper"])

        # Should not raise, and should return 0.0 for the failed paper
        assert results["error-paper"] == 0.0


class TestProcessBatchLlm:
    @pytest.mark.asyncio
    async def test_batch_llm_respects_budget(self, orch_config):
        campaign_id = _seed_campaign(orch_config, "domain_scan", llm_budget=5)

        # Set papers_llm_analyzed to be at the budget limit
        with get_session() as session:
            campaign = session.get(Campaign, campaign_id)
            campaign.papers_llm_analyzed = 5  # equals llm_budget

            session.add(
                Paper(
                    id="llm-paper",
                    title="LLM Paper",
                    source="test",
                    status="pending",
                )
            )
            session.add(
                CampaignPaper(
                    campaign_id=campaign_id,
                    paper_id="llm-paper",
                    source="domain_scan",
                    depth=0,
                    triage_status="llm_queued",
                )
            )

        with patch("snoopy.campaign.orchestrator.TriagePipeline") as MockTriage:
            mock_triage = MockTriage.return_value
            mock_triage.run_llm_tier = AsyncMock()

            orchestrator = CampaignOrchestrator(orch_config, campaign_id)
            orchestrator.triage = mock_triage
            await orchestrator._process_batch_llm(["llm-paper"])

            # Budget is 0, so LLM tier should not be called
            mock_triage.run_llm_tier.assert_not_called()


class TestSetStatus:
    @pytest.mark.asyncio
    async def test_set_status_updates_db(self, orch_config):
        campaign_id = _seed_campaign(orch_config, "domain_scan")

        orchestrator = CampaignOrchestrator(orch_config, campaign_id)
        await orchestrator._set_status("auto_analyzing")

        with get_session() as session:
            campaign = session.get(Campaign, campaign_id)
            assert campaign.status == "auto_analyzing"


class TestGetPapersAtDepth:
    @pytest.mark.asyncio
    async def test_returns_papers_at_correct_depth(self, orch_config):
        campaign_id = _seed_campaign(orch_config, "network_expansion")

        with get_session() as session:
            for i in range(3):
                pid = f"depth-paper-{i}"
                session.add(
                    Paper(
                        id=pid,
                        title=f"Paper {i}",
                        source="test",
                        status="pending",
                    )
                )
                session.add(
                    CampaignPaper(
                        campaign_id=campaign_id,
                        paper_id=pid,
                        source="seed",
                        depth=i,
                        triage_status="pending",
                    )
                )

        orchestrator = CampaignOrchestrator(orch_config, campaign_id)
        papers = await orchestrator._get_papers_at_depth(0, status="pending")
        assert papers == ["depth-paper-0"]

        papers = await orchestrator._get_papers_at_depth(1, status="pending")
        assert papers == ["depth-paper-1"]


class TestGetPromotedPapers:
    @pytest.mark.asyncio
    async def test_returns_papers_above_threshold(self, orch_config):
        campaign_id = _seed_campaign(orch_config, "domain_scan")

        with get_session() as session:
            # Paper above threshold
            session.add(Paper(id="high-score", title="High", source="test", status="pending"))
            session.add(
                CampaignPaper(
                    campaign_id=campaign_id,
                    paper_id="high-score",
                    source="domain_scan",
                    depth=0,
                    triage_status="auto_done",
                    auto_risk_score=50.0,
                )
            )

            # Paper below threshold
            session.add(Paper(id="low-score", title="Low", source="test", status="pending"))
            session.add(
                CampaignPaper(
                    campaign_id=campaign_id,
                    paper_id="low-score",
                    source="domain_scan",
                    depth=0,
                    triage_status="auto_done",
                    auto_risk_score=10.0,
                )
            )

        orchestrator = CampaignOrchestrator(orch_config, campaign_id)
        promoted = await orchestrator._get_promoted_papers()

        assert "high-score" in promoted
        assert "low-score" not in promoted

        # Verify promoted paper got marked as llm_queued
        with get_session() as session:
            from sqlalchemy import select

            cp = (
                session.execute(
                    select(CampaignPaper)
                    .where(CampaignPaper.campaign_id == campaign_id)
                    .where(CampaignPaper.paper_id == "high-score")
                )
                .scalars()
                .first()
            )
            assert cp.triage_status == "llm_queued"
            assert cp.llm_promoted is True
