"""Two-tier triage pipeline: cheap automated forensics first, LLM only on hits."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import func, select

from snoopy.config import SnoopyConfig
from snoopy.db.models import Campaign, CampaignPaper, Figure, Finding, Paper
from snoopy.db.session import get_async_session
from snoopy.pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)

# Auto-tier stages (cheap, high throughput)
AUTO_TIER_STAGES = [
    "download",
    "extract_text",
    "extract_figures",
    "extract_stats",
    "analyze_images_auto",
    "analyze_stats",
]

# LLM-tier stages (expensive, low throughput)
LLM_TIER_STAGES = [
    "classify_figures",
    "analyze_images_llm",
    "aggregate",
    "report",
]

# Risk score contributions (additive, capped at 100)
RISK_WEIGHTS: dict[str, float] = {
    "hash_match": 30.0,
    "clone_detection": 25.0,
    "grim": 15.0,
    "pvalue_check": 15.0,
    "retraction": 20.0,
    "ela": 10.0,
    "noise_analysis": 10.0,
    "pubpeer": 10.0,
    "benford": 5.0,
}


class TriagePipeline:
    """Two-tier funnel: automated forensics first, LLM only on pre-filtered hits."""

    def __init__(self, config: SnoopyConfig, campaign_id: str):
        self.config = config
        self.campaign_id = campaign_id
        self.orchestrator = PipelineOrchestrator(config)
        self.promotion_threshold = config.campaign.auto_risk_promotion_threshold

    async def run_auto_tier(self, paper_id: str) -> float:
        """Run tier 1 (automated forensics) on a paper. Returns auto_risk_score."""
        await self.orchestrator.process_paper_stages(paper_id, force_stages=AUTO_TIER_STAGES)

        # Check external signals (PubPeer, retraction status)
        await self._check_external_signals(paper_id)

        # Compute risk score from findings
        score = await self._compute_auto_risk_score(paper_id)

        # Update campaign paper record
        async with get_async_session() as session:
            result = await session.execute(
                select(CampaignPaper)
                .where(CampaignPaper.campaign_id == self.campaign_id)
                .where(CampaignPaper.paper_id == paper_id)
            )
            cp = result.scalars().first()
            if cp:
                cp.auto_risk_score = score  # type: ignore[assignment]
                cp.triage_status = "auto_done"  # type: ignore[assignment]

                # Count auto findings
                findings_result = await session.execute(
                    select(func.count(Finding.id)).where(Finding.paper_id == paper_id)
                )
                cp.auto_findings_count = findings_result.scalar() or 0  # type: ignore[assignment]

            # Update campaign counter
            campaign = await session.get(Campaign, self.campaign_id)
            if campaign:
                campaign.papers_triaged = (campaign.papers_triaged or 0) + 1  # type: ignore[assignment]
                campaign.updated_at = datetime.now(timezone.utc)  # type: ignore[assignment]

        return score

    async def run_llm_tier(self, paper_id: str) -> None:
        """Run tier 2 (LLM analysis) on a promoted paper."""
        async with get_async_session() as session:
            result = await session.execute(
                select(CampaignPaper)
                .where(CampaignPaper.campaign_id == self.campaign_id)
                .where(CampaignPaper.paper_id == paper_id)
            )
            cp = result.scalars().first()
            if cp:
                cp.triage_status = "llm_analyzing"  # type: ignore[assignment]

        await self.orchestrator.process_paper_stages(paper_id, force_stages=LLM_TIER_STAGES)

        async with get_async_session() as session:
            result = await session.execute(
                select(CampaignPaper)
                .where(CampaignPaper.campaign_id == self.campaign_id)
                .where(CampaignPaper.paper_id == paper_id)
            )
            cp = result.scalars().first()
            if cp:
                cp.triage_status = "complete"  # type: ignore[assignment]

                # Determine final risk from report
                from snoopy.db.models import Report

                report_result = await session.execute(
                    select(Report)
                    .where(Report.paper_id == paper_id)
                    .order_by(Report.created_at.desc())
                )
                report = report_result.scalars().first()
                if report:
                    cp.final_risk = str(report.overall_risk)  # type: ignore[assignment]

            # Update campaign counter
            campaign = await session.get(Campaign, self.campaign_id)
            if campaign:
                campaign.papers_llm_analyzed = (campaign.papers_llm_analyzed or 0) + 1  # type: ignore[assignment]
                campaign.updated_at = datetime.now(timezone.utc)  # type: ignore[assignment]

    async def run_paper_through_funnel(self, paper_id: str) -> str:
        """Complete funnel for a paper. Returns final triage_status."""
        # Mark as auto_analyzing
        async with get_async_session() as session:
            result = await session.execute(
                select(CampaignPaper)
                .where(CampaignPaper.campaign_id == self.campaign_id)
                .where(CampaignPaper.paper_id == paper_id)
            )
            cp = result.scalars().first()
            if cp:
                cp.triage_status = "auto_analyzing"  # type: ignore[assignment]

        # Run auto tier
        score = await self.run_auto_tier(paper_id)

        # Check promotion criteria
        should_promote = await self._should_promote(paper_id, score)

        if should_promote:
            async with get_async_session() as session:
                result = await session.execute(
                    select(CampaignPaper)
                    .where(CampaignPaper.campaign_id == self.campaign_id)
                    .where(CampaignPaper.paper_id == paper_id)
                )
                cp = result.scalars().first()
                if cp:
                    cp.llm_promoted = True  # type: ignore[assignment]
                    cp.triage_status = "llm_queued"  # type: ignore[assignment]

            await self.run_llm_tier(paper_id)
            return "complete"
        else:
            # Mark as dismissed or complete without LLM
            async with get_async_session() as session:
                result = await session.execute(
                    select(CampaignPaper)
                    .where(CampaignPaper.campaign_id == self.campaign_id)
                    .where(CampaignPaper.paper_id == paper_id)
                )
                cp = result.scalars().first()
                if cp:
                    if score < self.promotion_threshold:
                        cp.triage_status = "dismissed"  # type: ignore[assignment]
                        cp.final_risk = "clean"  # type: ignore[assignment]
                    else:
                        cp.triage_status = "complete"  # type: ignore[assignment]
            return "dismissed" if score < self.promotion_threshold else "complete"

    async def _compute_auto_risk_score(self, paper_id: str) -> float:
        """Compute additive risk score (0-100) from automated findings."""
        score = 0.0

        async with get_async_session() as session:
            # Get findings
            result = await session.execute(select(Finding).where(Finding.paper_id == paper_id))
            findings = result.scalars().all()

            # Track which risk categories have been triggered
            triggered: set[str] = set()
            for f in findings:
                analysis_type = str(f.analysis_type)
                if analysis_type in RISK_WEIGHTS and analysis_type not in triggered:
                    score += RISK_WEIGHTS[analysis_type]
                    triggered.add(analysis_type)

            # Check for cross-paper hash matches
            from snoopy.db.models import ImageHashMatch

            hash_result = await session.execute(
                select(func.count(ImageHashMatch.id)).where(
                    (ImageHashMatch.paper_id_a == paper_id)
                    | (ImageHashMatch.paper_id_b == paper_id)
                )
            )
            if (hash_result.scalar() or 0) > 0 and "hash_match" not in triggered:
                score += RISK_WEIGHTS["hash_match"]
                triggered.add("hash_match")

            # Check retraction status
            paper = await session.get(Paper, paper_id)
            if paper:
                if paper.retraction_status in ("retracted", "expression_of_concern"):
                    if "retraction" not in triggered:
                        score += RISK_WEIGHTS["retraction"]
                        triggered.add("retraction")
                if (paper.pubpeer_comments or 0) > 0:
                    if "pubpeer" not in triggered:
                        score += RISK_WEIGHTS["pubpeer"]
                        triggered.add("pubpeer")

        return min(score, 100.0)

    async def _should_promote(self, paper_id: str, score: float) -> bool:
        """Check whether a paper should be promoted to LLM tier."""
        if score < self.promotion_threshold:
            return False

        # Check LLM budget
        async with get_async_session() as session:
            campaign = await session.get(Campaign, self.campaign_id)
            if not campaign:
                return False
            if (campaign.papers_llm_analyzed or 0) >= (campaign.llm_budget or 100):
                return False

            # Check paper has at least 1 extracted figure
            fig_count = await session.execute(
                select(func.count(Figure.id)).where(Figure.paper_id == paper_id)
            )
            if (fig_count.scalar() or 0) == 0:
                return False

        return True

    async def _check_external_signals(self, paper_id: str) -> None:
        """Check PubPeer and retraction status for a paper."""
        async with get_async_session() as session:
            paper = await session.get(Paper, paper_id)
            if not paper or not paper.doi:
                return

            doi = str(paper.doi)

            # Check PubPeer (only if not already checked)
            if paper.pubpeer_comments is None:
                try:
                    from snoopy.discovery.pubpeer import check_pubpeer

                    pp_result = await check_pubpeer(doi)
                    paper.pubpeer_comments = pp_result.total_comments  # type: ignore[assignment]
                except Exception as e:
                    logger.debug("PubPeer check failed for %s: %s", doi, e)

            # Check retraction status (only if not already checked)
            if paper.retraction_status is None:
                try:
                    from snoopy.discovery.retraction_watch import check_retraction_status

                    retraction = await check_retraction_status(doi)
                    if retraction.is_retracted:
                        paper.retraction_status = "retracted"  # type: ignore[assignment]
                        paper.retraction_reason = retraction.retraction_reason  # type: ignore[assignment]
                    elif retraction.has_expression_of_concern:
                        paper.retraction_status = "expression_of_concern"  # type: ignore[assignment]
                except Exception as e:
                    logger.debug("Retraction check failed for %s: %s", doi, e)
