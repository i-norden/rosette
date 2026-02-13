"""Top-level campaign orchestrator with three investigation modes."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select

from snoopy.campaign.expander import NetworkExpander
from snoopy.campaign.hash_scanner import CampaignHashScanner
from snoopy.campaign.triage import TriagePipeline
from snoopy.config import SnoopyConfig
from snoopy.db.models import Campaign, CampaignPaper, Paper
from snoopy.db.session import get_async_session

logger = logging.getLogger(__name__)


class CampaignOrchestrator:
    """Top-level coordinator for campaign investigations.

    Supports three modes:
    - network_expansion: Follow co-author networks from seed papers
    - domain_scan: Systematic scan of a research domain
    - paper_mill: Detect paper mill clusters via image reuse
    """

    def __init__(self, config: SnoopyConfig, campaign_id: str):
        self.config = config
        self.campaign_id = campaign_id
        self.triage = TriagePipeline(config, campaign_id)
        self.expander = NetworkExpander(config, campaign_id)
        self.hash_scanner = CampaignHashScanner(config, campaign_id)
        self._paused = False

    async def run(self) -> None:
        """Execute campaign based on mode."""
        async with get_async_session() as session:
            campaign = await session.get(Campaign, self.campaign_id)
            if not campaign:
                raise ValueError(f"Campaign {self.campaign_id} not found")

            mode = str(campaign.mode)
            status = str(campaign.status)

            if status == "completed":
                logger.info("Campaign %s already completed", self.campaign_id)
                return
            if status == "paused":
                logger.info("Resuming campaign %s", self.campaign_id)

        self._paused = False

        if mode == "network_expansion":
            await self._run_network_expansion()
        elif mode == "domain_scan":
            await self._run_domain_scan()
        elif mode == "paper_mill":
            await self._run_paper_mill()
        else:
            raise ValueError(f"Unknown campaign mode: {mode}")

        if not self._paused:
            await self._set_status("completed")

    async def pause(self) -> None:
        """Pause the campaign (active tasks will complete)."""
        self._paused = True
        await self._set_status("paused")
        logger.info("Campaign %s paused", self.campaign_id)

    async def resume(self) -> None:
        """Resume a paused campaign."""
        await self.run()

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------

    async def _run_network_expansion(self) -> None:
        """Network expansion mode: iterative depth-based co-author traversal."""
        async with get_async_session() as session:
            campaign = await session.get(Campaign, self.campaign_id)
            if not campaign:
                return
            max_depth = campaign.max_depth or 2

        # Seed phase
        await self._set_status("seeding")
        await self._seed_papers()

        for depth in range(max_depth + 1):
            if self._paused:
                break

            # Process all papers at current depth through auto-tier
            await self._set_status("auto_analyzing")
            paper_ids = await self._get_papers_at_depth(depth, status="pending")
            if paper_ids:
                await self._process_batch_auto(paper_ids)

            if self._paused:
                break

            # Run hash scanner on new figures
            await self.hash_scanner.scan_incremental(paper_ids)

            # Promote and process LLM tier
            await self._set_status("llm_analyzing")
            promoted_ids = await self._get_promoted_papers()
            if promoted_ids:
                await self._process_batch_llm(promoted_ids)

            if self._paused:
                break

            # Expand network from flagged papers
            if depth < max_depth:
                await self._set_status("expanding")
                added = await self.expander.expand_depth(depth + 1)
                logger.info("Depth %d: expanded %d new papers", depth, added)

        # Final: run author network analysis
        if not self._paused:
            await self._run_author_network_analysis()

    async def _run_domain_scan(self) -> None:
        """Domain scan mode: systematic scan of a research field/journal."""
        await self._set_status("seeding")

        # Discover papers via OpenAlex
        async with get_async_session() as session:
            campaign = await session.get(Campaign, self.campaign_id)
            if not campaign:
                return
            config_data = json.loads(str(campaign.config_json or "{}"))

        field = config_data.get("field", "biomedical")
        min_citations = config_data.get("min_citations", 50)
        max_papers = int(
            config_data.get("max_papers") or (campaign.max_papers if campaign else 1000)
        )

        from snoopy.discovery.openalex import search_works

        works = await search_works(
            query=field,
            field=field,
            min_citations=min_citations,
            limit=max_papers,
        )

        # Create paper and campaign_paper records
        async with get_async_session() as session:
            campaign = await session.get(Campaign, self.campaign_id)
            if not campaign:
                return

            for work in works:
                if self._paused:
                    break

                paper_id = str(uuid.uuid4())

                # Check if DOI already exists
                if work.doi:
                    existing = await session.execute(select(Paper).where(Paper.doi == work.doi))
                    existing_paper = existing.scalars().first()
                    if existing_paper:
                        paper_id = str(existing_paper.id)
                    else:
                        new_paper = Paper(
                            id=paper_id,
                            doi=work.doi,
                            title=work.title or "Unknown",
                            authors_json=json.dumps([a.name for a in work.authors]),
                            journal=work.journal,
                            citation_count=work.citation_count,
                            publication_year=work.publication_year,
                            source="campaign_domain_scan",
                            status="pending",
                        )
                        session.add(new_paper)
                else:
                    new_paper = Paper(
                        id=paper_id,
                        title=work.title or "Unknown",
                        authors_json=json.dumps([a.name for a in work.authors]),
                        journal=work.journal,
                        citation_count=work.citation_count,
                        publication_year=work.publication_year,
                        source="campaign_domain_scan",
                        status="pending",
                    )
                    session.add(new_paper)

                # Check if already in campaign
                cp_check = await session.execute(
                    select(CampaignPaper)
                    .where(CampaignPaper.campaign_id == self.campaign_id)
                    .where(CampaignPaper.paper_id == paper_id)
                )
                if not cp_check.scalars().first():
                    cp = CampaignPaper(
                        campaign_id=self.campaign_id,
                        paper_id=paper_id,
                        source="domain_scan",
                        depth=0,
                        triage_status="pending",
                    )
                    session.add(cp)
                    campaign.papers_discovered = (campaign.papers_discovered or 0) + 1  # type: ignore[assignment]

            campaign.updated_at = datetime.now(timezone.utc)  # type: ignore[assignment]

        if self._paused:
            return

        # Process all through auto-tier
        await self._set_status("auto_analyzing")
        all_paper_ids = await self._get_papers_at_depth(0, status="pending")
        if all_paper_ids:
            await self._process_batch_auto(all_paper_ids)

        if self._paused:
            return

        # Run hash scanner across all figures
        await self.hash_scanner.scan_all_pairs()

        # Promote and process LLM tier
        await self._set_status("llm_analyzing")
        promoted_ids = await self._get_promoted_papers()
        if promoted_ids:
            await self._process_batch_llm(promoted_ids)

    async def _run_paper_mill(self) -> None:
        """Paper mill mode: follow image reuse connectivity."""
        await self._set_status("seeding")
        await self._seed_papers()

        iteration = 0
        max_iterations = 50  # safety cap

        while iteration < max_iterations and not self._paused:
            iteration += 1

            # Process pending papers through auto-tier
            await self._set_status("auto_analyzing")
            pending = await self._get_pending_papers()
            if not pending:
                break

            await self._process_batch_auto(pending)

            if self._paused:
                break

            # Run hash scanner
            matches = await self.hash_scanner.scan_incremental(pending)

            # For any hash match, add matched paper to campaign
            new_paper_ids: list[str] = []
            for match in matches:
                matched_ids = [str(match.paper_id_a), str(match.paper_id_b)]
                for pid in matched_ids:
                    async with get_async_session() as session:
                        # Check if paper is already in campaign
                        cp_check = await session.execute(
                            select(CampaignPaper)
                            .where(CampaignPaper.campaign_id == self.campaign_id)
                            .where(CampaignPaper.paper_id == pid)
                        )
                        if not cp_check.scalars().first():
                            cp = CampaignPaper(
                                campaign_id=self.campaign_id,
                                paper_id=pid,
                                source="hash_match",
                                depth=iteration,
                                triage_status="pending",
                            )
                            session.add(cp)
                            new_paper_ids.append(pid)

                            campaign = await session.get(Campaign, self.campaign_id)
                            if campaign:
                                campaign.papers_discovered = (campaign.papers_discovered or 0) + 1  # type: ignore[assignment]

            # Also expand to co-authors of matched papers
            for pid in new_paper_ids:
                if self._paused:
                    break
                await self.expander.expand_from_paper(pid, iteration)

            # If no new papers found, we're done
            if not new_paper_ids:
                break

        # Promote and process LLM tier
        if not self._paused:
            await self._set_status("llm_analyzing")
            promoted_ids = await self._get_promoted_papers()
            if promoted_ids:
                await self._process_batch_llm(promoted_ids)

            await self._run_author_network_analysis()

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    async def _seed_papers(self) -> None:
        """Create Paper and CampaignPaper records for seed DOIs."""
        async with get_async_session() as session:
            campaign = await session.get(Campaign, self.campaign_id)
            if not campaign or not campaign.seed_dois:
                return

            try:
                seed_dois = json.loads(str(campaign.seed_dois))
            except (json.JSONDecodeError, TypeError):
                return

            for doi in seed_dois:
                paper_id = str(uuid.uuid4())

                # Check if DOI already exists
                existing = await session.execute(select(Paper).where(Paper.doi == doi))
                existing_paper = existing.scalars().first()
                if existing_paper:
                    paper_id = str(existing_paper.id)
                else:
                    new_paper = Paper(
                        id=paper_id,
                        doi=doi,
                        title=doi,  # Will be updated during processing
                        source="campaign_seed",
                        status="pending",
                    )
                    session.add(new_paper)

                # Check if already in campaign
                cp_check = await session.execute(
                    select(CampaignPaper)
                    .where(CampaignPaper.campaign_id == self.campaign_id)
                    .where(CampaignPaper.paper_id == paper_id)
                )
                if not cp_check.scalars().first():
                    cp = CampaignPaper(
                        campaign_id=self.campaign_id,
                        paper_id=paper_id,
                        source="seed",
                        depth=0,
                        triage_status="pending",
                    )
                    session.add(cp)
                    campaign.papers_discovered = (campaign.papers_discovered or 0) + 1  # type: ignore[assignment]

            campaign.updated_at = datetime.now(timezone.utc)  # type: ignore[assignment]

    async def _process_batch_auto(self, paper_ids: list[str]) -> dict[str, float]:
        """Process papers through auto-tier with bounded concurrency."""
        sem = asyncio.Semaphore(self.config.campaign.batch_concurrency)
        results: dict[str, float] = {}

        async def _process_one(pid: str) -> None:
            if self._paused:
                return
            async with sem:
                try:
                    score = await self.triage.run_auto_tier(pid)
                    results[pid] = score
                except Exception as e:
                    logger.error("Auto-tier failed for paper %s: %s", pid, e)
                    results[pid] = 0.0

        tasks = [_process_one(pid) for pid in paper_ids]
        await asyncio.gather(*tasks)

        # Update campaign flagged count
        async with get_async_session() as session:
            campaign = await session.get(Campaign, self.campaign_id)
            if campaign:
                flagged = sum(
                    1
                    for s in results.values()
                    if s >= self.config.campaign.auto_risk_promotion_threshold
                )
                campaign.papers_flagged = (campaign.papers_flagged or 0) + flagged  # type: ignore[assignment]
                campaign.updated_at = datetime.now(timezone.utc)  # type: ignore[assignment]

        return results

    async def _process_batch_llm(self, paper_ids: list[str]) -> None:
        """Process papers through LLM-tier with budget tracking."""
        sem = asyncio.Semaphore(self.config.campaign.batch_concurrency)

        async def _process_one(pid: str) -> None:
            if self._paused:
                return
            # Check budget before each paper
            async with get_async_session() as session:
                campaign = await session.get(Campaign, self.campaign_id)
                if not campaign:
                    return
                if (campaign.papers_llm_analyzed or 0) >= (campaign.llm_budget or 100):
                    return

            async with sem:
                try:
                    await self.triage.run_llm_tier(pid)
                except Exception as e:
                    logger.error("LLM-tier failed for paper %s: %s", pid, e)

        tasks = [_process_one(pid) for pid in paper_ids]
        await asyncio.gather(*tasks)

    async def _get_papers_at_depth(self, depth: int, status: str = "pending") -> list[str]:
        """Get paper IDs at a specific depth with given triage status."""
        async with get_async_session() as session:
            result = await session.execute(
                select(CampaignPaper.paper_id)
                .where(CampaignPaper.campaign_id == self.campaign_id)
                .where(CampaignPaper.depth == depth)
                .where(CampaignPaper.triage_status == status)
            )
            return [str(row[0]) for row in result.all()]

    async def _get_pending_papers(self) -> list[str]:
        """Get all pending paper IDs in the campaign."""
        async with get_async_session() as session:
            result = await session.execute(
                select(CampaignPaper.paper_id)
                .where(CampaignPaper.campaign_id == self.campaign_id)
                .where(CampaignPaper.triage_status == "pending")
            )
            return [str(row[0]) for row in result.all()]

    async def _get_promoted_papers(self) -> list[str]:
        """Get paper IDs that should be promoted to LLM tier."""
        async with get_async_session() as session:
            threshold = self.config.campaign.auto_risk_promotion_threshold
            result = await session.execute(
                select(CampaignPaper.paper_id)
                .where(CampaignPaper.campaign_id == self.campaign_id)
                .where(CampaignPaper.triage_status == "auto_done")
                .where(CampaignPaper.auto_risk_score >= threshold)
            )
            paper_ids = [str(row[0]) for row in result.all()]

            # Mark as llm_queued
            for pid in paper_ids:
                cp_result = await session.execute(
                    select(CampaignPaper)
                    .where(CampaignPaper.campaign_id == self.campaign_id)
                    .where(CampaignPaper.paper_id == pid)
                )
                cp = cp_result.scalars().first()
                if cp:
                    cp.llm_promoted = True  # type: ignore[assignment]
                    cp.triage_status = "llm_queued"  # type: ignore[assignment]

            return paper_ids

    async def _set_status(self, status: str) -> None:
        """Update campaign status."""
        async with get_async_session() as session:
            campaign = await session.get(Campaign, self.campaign_id)
            if campaign:
                campaign.status = status  # type: ignore[assignment]
                campaign.updated_at = datetime.now(timezone.utc)  # type: ignore[assignment]

    async def _run_author_network_analysis(self) -> None:
        """Run Louvain community detection on co-author graph."""
        try:
            from snoopy.analysis.author_network import run_network_analysis

            result = run_network_analysis()
            logger.info(
                "Author network: %d authors, %d clusters, %d high-risk authors",
                result.total_authors,
                result.total_communities,
                len(result.high_risk_authors),
            )
        except Exception as e:
            logger.warning("Author network analysis failed: %s", e)
