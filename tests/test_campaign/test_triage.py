"""Tests for snoopy.campaign.triage module.

Tests the two-tier triage funnel: auto risk scoring, promotion logic,
and LLM tier execution with mocked pipeline stages.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from snoopy.campaign.triage import RISK_WEIGHTS, TriagePipeline
from snoopy.config import SnoopyConfig
from snoopy.db.models import (
    Campaign,
    CampaignPaper,
    Figure,
    Finding,
    ImageHashMatch,
    Paper,
    Report,
)
from snoopy.db.session import get_session, init_async_db, init_db


@pytest.fixture
def campaign_config(tmp_path) -> SnoopyConfig:
    db_path = tmp_path / "triage_test.db"
    return SnoopyConfig(
        storage={
            "database_url": f"sqlite:///{db_path}",
            "pdf_dir": str(tmp_path / "pdfs"),
            "figures_dir": str(tmp_path / "figures"),
            "reports_dir": str(tmp_path / "reports"),
        },
        campaign={"auto_risk_promotion_threshold": 30.0},
    )


@pytest.fixture
def seeded_campaign(campaign_config) -> tuple[str, str]:
    """Initialize DB, create a campaign and a paper. Return (campaign_id, paper_id)."""
    init_db(campaign_config.storage.database_url)
    init_async_db(campaign_config.storage.database_url)

    campaign_id = "test-campaign-triage-001"
    paper_id = "test-paper-triage-001"

    with get_session() as session:
        campaign = Campaign(
            id=campaign_id,
            name="Triage Test Campaign",
            mode="domain_scan",
            status="auto_analyzing",
            max_papers=100,
            llm_budget=10,
            papers_discovered=1,
        )
        session.add(campaign)

        paper = Paper(
            id=paper_id,
            title="Suspicious Paper",
            doi="10.1234/triage.test",
            source="test",
            status="pending",
        )
        session.add(paper)

        cp = CampaignPaper(
            campaign_id=campaign_id,
            paper_id=paper_id,
            source="domain_scan",
            depth=0,
            triage_status="pending",
        )
        session.add(cp)

    return campaign_id, paper_id


class TestRiskWeights:
    def test_all_expected_keys_present(self):
        expected = {
            "hash_match", "clone_detection", "grim", "pvalue_check",
            "retraction", "ela", "noise_analysis", "pubpeer", "benford",
            "tortured_phrases", "grimmer", "variance_ratio", "sprite",
            "dct_analysis", "jpeg_ghost", "fft_analysis", "terminal_digit",
            "temporal_patterns",
        }
        assert set(RISK_WEIGHTS.keys()) == expected

    def test_total_possible_score_exceeds_100(self):
        """Total is >100, but scoring is capped at 100."""
        total = sum(RISK_WEIGHTS.values())
        assert total > 100


class TestComputeAutoRiskScore:
    @pytest.mark.asyncio
    async def test_no_findings_returns_zero(self, campaign_config, seeded_campaign):
        campaign_id, paper_id = seeded_campaign
        triage = TriagePipeline(campaign_config, campaign_id)
        score = await triage._compute_auto_risk_score(paper_id)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_single_finding_contributes_weight(self, campaign_config, seeded_campaign):
        campaign_id, paper_id = seeded_campaign

        with get_session() as session:
            finding = Finding(
                paper_id=paper_id,
                analysis_type="clone_detection",
                severity="high",
                confidence=0.9,
                title="Clone detected",
            )
            session.add(finding)

        triage = TriagePipeline(campaign_config, campaign_id)
        score = await triage._compute_auto_risk_score(paper_id)
        assert score == RISK_WEIGHTS["clone_detection"]

    @pytest.mark.asyncio
    async def test_multiple_findings_additive(self, campaign_config, seeded_campaign):
        campaign_id, paper_id = seeded_campaign

        with get_session() as session:
            session.add(Finding(
                paper_id=paper_id, analysis_type="clone_detection",
                severity="high", confidence=0.9, title="Clone",
            ))
            session.add(Finding(
                paper_id=paper_id, analysis_type="grim",
                severity="medium", confidence=0.7, title="GRIM fail",
            ))

        triage = TriagePipeline(campaign_config, campaign_id)
        score = await triage._compute_auto_risk_score(paper_id)
        expected = RISK_WEIGHTS["clone_detection"] + RISK_WEIGHTS["grim"]
        assert score == expected

    @pytest.mark.asyncio
    async def test_duplicate_analysis_type_counted_once(self, campaign_config, seeded_campaign):
        campaign_id, paper_id = seeded_campaign

        with get_session() as session:
            session.add(Finding(
                paper_id=paper_id, analysis_type="ela",
                severity="medium", confidence=0.8, title="ELA anomaly 1",
            ))
            session.add(Finding(
                paper_id=paper_id, analysis_type="ela",
                severity="low", confidence=0.5, title="ELA anomaly 2",
            ))

        triage = TriagePipeline(campaign_config, campaign_id)
        score = await triage._compute_auto_risk_score(paper_id)
        assert score == RISK_WEIGHTS["ela"]

    @pytest.mark.asyncio
    async def test_score_capped_at_100(self, campaign_config, seeded_campaign):
        campaign_id, paper_id = seeded_campaign

        # Add findings for every risk category
        with get_session() as session:
            for analysis_type in RISK_WEIGHTS:
                if analysis_type in ("hash_match", "retraction", "pubpeer"):
                    continue  # These come from other sources
                session.add(Finding(
                    paper_id=paper_id, analysis_type=analysis_type,
                    severity="high", confidence=0.9, title=f"{analysis_type} hit",
                ))

            # Add hash match
            fig_a = Figure(paper_id=paper_id, phash="abcd1234")
            session.add(fig_a)

        # Add hash match record
        other_paper_id = "other-paper-001"
        with get_session() as session:
            session.add(Paper(
                id=other_paper_id, title="Other Paper", source="test", status="pending",
            ))
            fig_b = Figure(paper_id=other_paper_id, phash="abcd1235")
            session.add(fig_b)

        with get_session() as session:
            from sqlalchemy import select
            fig_a_id = session.execute(
                select(Figure.id).where(Figure.paper_id == paper_id)
            ).scalar()
            fig_b_id = session.execute(
                select(Figure.id).where(Figure.paper_id == other_paper_id)
            ).scalar()

            session.add(ImageHashMatch(
                figure_id_a=fig_a_id, figure_id_b=fig_b_id,
                paper_id_a=paper_id, paper_id_b=other_paper_id,
                hash_type="phash", hash_distance=5,
            ))

            # Set retraction status
            paper = session.get(Paper, paper_id)
            paper.retraction_status = "retracted"
            paper.pubpeer_comments = 3

        triage = TriagePipeline(campaign_config, campaign_id)
        score = await triage._compute_auto_risk_score(paper_id)
        assert score == 100.0

    @pytest.mark.asyncio
    async def test_retraction_status_adds_score(self, campaign_config, seeded_campaign):
        campaign_id, paper_id = seeded_campaign

        with get_session() as session:
            paper = session.get(Paper, paper_id)
            paper.retraction_status = "retracted"

        triage = TriagePipeline(campaign_config, campaign_id)
        score = await triage._compute_auto_risk_score(paper_id)
        assert score == RISK_WEIGHTS["retraction"]

    @pytest.mark.asyncio
    async def test_pubpeer_comments_add_score(self, campaign_config, seeded_campaign):
        campaign_id, paper_id = seeded_campaign

        with get_session() as session:
            paper = session.get(Paper, paper_id)
            paper.pubpeer_comments = 5

        triage = TriagePipeline(campaign_config, campaign_id)
        score = await triage._compute_auto_risk_score(paper_id)
        assert score == RISK_WEIGHTS["pubpeer"]


class TestShouldPromote:
    @pytest.mark.asyncio
    async def test_low_score_not_promoted(self, campaign_config, seeded_campaign):
        campaign_id, paper_id = seeded_campaign

        # Add a figure so that check passes
        with get_session() as session:
            session.add(Figure(paper_id=paper_id, phash="abcd1234"))

        triage = TriagePipeline(campaign_config, campaign_id)
        result = await triage._should_promote(paper_id, 10.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_high_score_promoted(self, campaign_config, seeded_campaign):
        campaign_id, paper_id = seeded_campaign

        with get_session() as session:
            session.add(Figure(paper_id=paper_id, phash="abcd1234"))

        triage = TriagePipeline(campaign_config, campaign_id)
        result = await triage._should_promote(paper_id, 50.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_budget_exhausted_blocks_promotion(self, campaign_config, seeded_campaign):
        campaign_id, paper_id = seeded_campaign

        with get_session() as session:
            session.add(Figure(paper_id=paper_id, phash="abcd1234"))
            campaign = session.get(Campaign, campaign_id)
            campaign.papers_llm_analyzed = 10  # equals llm_budget
            campaign.llm_budget = 10

        triage = TriagePipeline(campaign_config, campaign_id)
        result = await triage._should_promote(paper_id, 50.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_no_figures_blocks_promotion(self, campaign_config, seeded_campaign):
        campaign_id, paper_id = seeded_campaign
        # No figures added

        triage = TriagePipeline(campaign_config, campaign_id)
        result = await triage._should_promote(paper_id, 50.0)
        assert result is False


class TestRunAutoTier:
    @pytest.mark.asyncio
    async def test_auto_tier_updates_campaign_paper(self, campaign_config, seeded_campaign):
        campaign_id, paper_id = seeded_campaign

        with patch.object(
            TriagePipeline, "_check_external_signals", new_callable=AsyncMock
        ):
            with patch(
                "snoopy.campaign.triage.PipelineOrchestrator"
            ) as MockOrch:
                mock_orch = MockOrch.return_value
                mock_orch.process_paper_stages = AsyncMock()

                triage = TriagePipeline(campaign_config, campaign_id)
                triage.orchestrator = mock_orch

                score = await triage.run_auto_tier(paper_id)

        assert score == 0.0  # No findings, so score is 0

        with get_session() as session:
            from sqlalchemy import select
            cp = session.execute(
                select(CampaignPaper)
                .where(CampaignPaper.campaign_id == campaign_id)
                .where(CampaignPaper.paper_id == paper_id)
            ).scalars().first()
            assert cp.triage_status == "auto_done"
            assert cp.auto_risk_score == 0.0

            campaign = session.get(Campaign, campaign_id)
            assert campaign.papers_triaged == 1


class TestRunLlmTier:
    @pytest.mark.asyncio
    async def test_llm_tier_marks_complete(self, campaign_config, seeded_campaign):
        campaign_id, paper_id = seeded_campaign

        # Set up paper with auto_done status
        with get_session() as session:
            from sqlalchemy import select
            cp = session.execute(
                select(CampaignPaper)
                .where(CampaignPaper.campaign_id == campaign_id)
                .where(CampaignPaper.paper_id == paper_id)
            ).scalars().first()
            cp.triage_status = "auto_done"
            cp.llm_promoted = True

        with patch(
            "snoopy.campaign.triage.PipelineOrchestrator"
        ) as MockOrch:
            mock_orch = MockOrch.return_value
            mock_orch.process_paper_stages = AsyncMock()

            triage = TriagePipeline(campaign_config, campaign_id)
            triage.orchestrator = mock_orch

            await triage.run_llm_tier(paper_id)

        with get_session() as session:
            from sqlalchemy import select
            cp = session.execute(
                select(CampaignPaper)
                .where(CampaignPaper.campaign_id == campaign_id)
                .where(CampaignPaper.paper_id == paper_id)
            ).scalars().first()
            assert cp.triage_status == "complete"

            campaign = session.get(Campaign, campaign_id)
            assert campaign.papers_llm_analyzed == 1

    @pytest.mark.asyncio
    async def test_llm_tier_picks_up_report_risk(self, campaign_config, seeded_campaign):
        campaign_id, paper_id = seeded_campaign

        # Create a report for the paper
        with get_session() as session:
            report = Report(
                paper_id=paper_id,
                overall_risk="high",
                overall_confidence=0.85,
                summary="Suspicious findings",
            )
            session.add(report)

        with patch(
            "snoopy.campaign.triage.PipelineOrchestrator"
        ) as MockOrch:
            mock_orch = MockOrch.return_value
            mock_orch.process_paper_stages = AsyncMock()

            triage = TriagePipeline(campaign_config, campaign_id)
            triage.orchestrator = mock_orch

            await triage.run_llm_tier(paper_id)

        with get_session() as session:
            from sqlalchemy import select
            cp = session.execute(
                select(CampaignPaper)
                .where(CampaignPaper.campaign_id == campaign_id)
                .where(CampaignPaper.paper_id == paper_id)
            ).scalars().first()
            assert cp.final_risk == "high"


class TestRunPaperThroughFunnel:
    @pytest.mark.asyncio
    async def test_low_score_dismissed(self, campaign_config, seeded_campaign):
        campaign_id, paper_id = seeded_campaign

        with patch.object(
            TriagePipeline, "run_auto_tier", new_callable=AsyncMock, return_value=5.0
        ):
            with patch.object(
                TriagePipeline, "_should_promote", new_callable=AsyncMock, return_value=False
            ):
                triage = TriagePipeline(campaign_config, campaign_id)
                status = await triage.run_paper_through_funnel(paper_id)

        assert status == "dismissed"

    @pytest.mark.asyncio
    async def test_high_score_promoted_and_completed(self, campaign_config, seeded_campaign):
        campaign_id, paper_id = seeded_campaign

        with patch.object(
            TriagePipeline, "run_auto_tier", new_callable=AsyncMock, return_value=50.0
        ):
            with patch.object(
                TriagePipeline, "_should_promote", new_callable=AsyncMock, return_value=True
            ):
                with patch.object(
                    TriagePipeline, "run_llm_tier", new_callable=AsyncMock
                ):
                    triage = TriagePipeline(campaign_config, campaign_id)
                    status = await triage.run_paper_through_funnel(paper_id)

        assert status == "complete"
