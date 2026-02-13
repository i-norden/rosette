"""Network expansion from flagged papers via co-author traversal."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select

from snoopy.config import SnoopyConfig
from snoopy.db.models import (
    Author,
    AuthorPaperLink,
    Campaign,
    CampaignPaper,
    Paper,
)
from snoopy.db.session import get_async_session

logger = logging.getLogger(__name__)


class NetworkExpander:
    """Expands campaign scope by following co-author networks from flagged papers."""

    def __init__(self, config: SnoopyConfig, campaign_id: str):
        self.config = config
        self.campaign_id = campaign_id
        self.max_authors_per_paper = config.campaign.max_authors_per_paper
        self.max_papers_per_author = config.campaign.max_papers_per_author

    async def expand_from_paper(
        self, paper_id: str, depth: int, source_paper_id: str | None = None
    ) -> list[str]:
        """Expand from a single paper by finding co-author publications.

        Returns list of newly added paper_ids.
        """
        new_paper_ids: list[str] = []

        async with get_async_session() as session:
            # Check campaign cap
            campaign = await session.get(Campaign, self.campaign_id)
            if not campaign:
                return []
            current_count = campaign.papers_discovered or 0
            max_papers = campaign.max_papers or 1000
            if current_count >= max_papers:
                logger.info("Campaign %s reached max_papers cap", self.campaign_id)
                return []

            # Get paper's authors
            paper = await session.get(Paper, paper_id)
            if not paper or not paper.authors_json:
                return []

            try:
                authors = json.loads(str(paper.authors_json))
            except (json.JSONDecodeError, TypeError):
                return []

            # Limit authors per paper
            if isinstance(authors, list):
                authors = authors[: self.max_authors_per_paper]

        # For each author, find their publications via OpenAlex
        for author_info in authors:
            author_name = (
                author_info if isinstance(author_info, str) else author_info.get("name", "")
            )
            if not author_name:
                continue

            try:
                candidates = await self._find_author_papers(author_name)
            except Exception as e:
                logger.debug("Failed to find papers for author %s: %s", author_name, e)
                continue

            # Check retraction history
            author_id = await self._ensure_author_record(author_name, author_info)
            await self.check_author_history(author_name, author_id)

            for candidate in candidates:
                async with get_async_session() as session:
                    campaign = await session.get(Campaign, self.campaign_id)
                    if not campaign:
                        break
                    if (campaign.papers_discovered or 0) >= (campaign.max_papers or 1000):
                        break

                    # Skip if already in campaign
                    existing = await session.execute(
                        select(CampaignPaper)
                        .where(CampaignPaper.campaign_id == self.campaign_id)
                        .where(CampaignPaper.paper_id == candidate["paper_id"])
                    )
                    if existing.scalars().first():
                        continue

                    # Also check by DOI
                    if candidate.get("doi"):
                        doi_check = await session.execute(
                            select(Paper).where(Paper.doi == candidate["doi"])
                        )
                        existing_paper = doi_check.scalars().first()
                        if existing_paper:
                            # Paper exists, check if in campaign
                            cp_check = await session.execute(
                                select(CampaignPaper)
                                .where(CampaignPaper.campaign_id == self.campaign_id)
                                .where(CampaignPaper.paper_id == str(existing_paper.id))
                            )
                            if cp_check.scalars().first():
                                continue
                            candidate["paper_id"] = str(existing_paper.id)
                        else:
                            # Create new paper record
                            new_paper = Paper(
                                id=candidate["paper_id"],
                                doi=candidate.get("doi"),
                                title=candidate.get("title", "Unknown"),
                                authors_json=json.dumps(candidate.get("authors", [])),
                                journal=candidate.get("journal"),
                                citation_count=candidate.get("citation_count", 0),
                                publication_year=candidate.get("publication_year"),
                                source="campaign_expansion",
                                status="pending",
                            )
                            session.add(new_paper)
                    else:
                        # Create paper without DOI
                        new_paper = Paper(
                            id=candidate["paper_id"],
                            title=candidate.get("title", "Unknown"),
                            authors_json=json.dumps(candidate.get("authors", [])),
                            journal=candidate.get("journal"),
                            citation_count=candidate.get("citation_count", 0),
                            publication_year=candidate.get("publication_year"),
                            source="campaign_expansion",
                            status="pending",
                        )
                        session.add(new_paper)

                    # Create campaign paper link
                    cp = CampaignPaper(
                        campaign_id=self.campaign_id,
                        paper_id=candidate["paper_id"],
                        source="network_expansion",
                        source_paper_id=source_paper_id or paper_id,
                        source_author_id=author_id,
                        depth=depth,
                        triage_status="pending",
                    )
                    session.add(cp)

                    # Update campaign counter
                    campaign.papers_discovered = (campaign.papers_discovered or 0) + 1  # type: ignore[assignment]
                    campaign.updated_at = datetime.now(timezone.utc)  # type: ignore[assignment]

                    new_paper_ids.append(candidate["paper_id"])

                    # Create author-paper link
                    link = AuthorPaperLink(
                        author_id=author_id,
                        paper_id=candidate["paper_id"],
                    )
                    session.add(link)

        return new_paper_ids

    async def expand_depth(self, target_depth: int) -> int:
        """Expand all flagged papers at depth-1. Returns count of papers added."""
        async with get_async_session() as session:
            result = await session.execute(
                select(CampaignPaper)
                .where(CampaignPaper.campaign_id == self.campaign_id)
                .where(CampaignPaper.depth == target_depth - 1)
                .where(CampaignPaper.final_risk.in_(["critical", "high", "medium"]))
            )
            flagged_papers = result.scalars().all()
            flagged_ids = [(str(cp.paper_id), str(cp.paper_id)) for cp in flagged_papers]

        total_added = 0
        for paper_id, source_paper_id in flagged_ids:
            new_ids = await self.expand_from_paper(paper_id, target_depth, source_paper_id)
            total_added += len(new_ids)

        logger.info(
            "Depth %d expansion: %d flagged papers -> %d new candidates",
            target_depth,
            len(flagged_ids),
            total_added,
        )
        return total_added

    async def check_author_history(self, author_name: str, author_id: str | None = None) -> dict:
        """Check retraction + PubPeer history for an author."""
        from snoopy.discovery.retraction_watch import check_author_retractions

        history = await check_author_retractions(author_name)

        if author_id:
            async with get_async_session() as session:
                author = await session.get(Author, author_id)
                if author:
                    author.retraction_count = history.total_retractions  # type: ignore[assignment]
                    author.updated_at = datetime.now(timezone.utc)  # type: ignore[assignment]

        return {
            "author_name": author_name,
            "total_retractions": history.total_retractions,
            "total_expressions_of_concern": history.total_expressions_of_concern,
            "retracted_dois": history.retracted_dois,
        }

    async def _find_author_papers(self, author_name: str) -> list[dict]:
        """Find papers by an author using OpenAlex."""
        from snoopy.discovery.openalex import search_works

        works = await search_works(
            query=author_name,
            limit=self.max_papers_per_author,
        )

        candidates = []
        for work in works:
            paper_id = str(uuid.uuid4())
            authors_list = [{"name": a.name, "openalex_id": a.openalex_id} for a in work.authors]
            candidates.append(
                {
                    "paper_id": paper_id,
                    "doi": work.doi,
                    "title": work.title,
                    "authors": authors_list,
                    "journal": work.journal,
                    "citation_count": work.citation_count,
                    "publication_year": work.publication_year,
                }
            )

        return candidates

    async def _ensure_author_record(self, author_name: str, author_info: dict | str) -> str:
        """Get or create an Author record, returning the author ID."""
        async with get_async_session() as session:
            # Try to find existing author by name
            result = await session.execute(select(Author).where(Author.name == author_name))
            author = result.scalars().first()
            if author:
                return str(author.id)

            # Create new author
            author_id = str(uuid.uuid4())
            orcid = None
            institution = None
            if isinstance(author_info, dict):
                orcid = author_info.get("orcid")
                institution = author_info.get("institution")
                if isinstance(institution, dict):
                    institution = institution.get("name") or institution.get("display_name")

            new_author = Author(
                id=author_id,
                name=author_name,
                orcid=orcid,
                institution=institution,
            )
            session.add(new_author)
            return author_id
