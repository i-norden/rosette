"""Tests for rosette.campaign.expander module.

Tests network expansion from flagged papers via co-author traversal,
with mocked OpenAlex and retraction watch APIs.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rosette.campaign.expander import NetworkExpander
from rosette.config import RosetteConfig
from rosette.db.models import Author, Campaign, CampaignPaper, Paper
from rosette.db.session import get_session, init_async_db, init_db


@pytest.fixture
def expander_config(tmp_path) -> RosetteConfig:
    db_path = tmp_path / "expander_test.db"
    return RosetteConfig(
        storage={
            "database_url": f"sqlite:///{db_path}",
            "pdf_dir": str(tmp_path / "pdfs"),
            "figures_dir": str(tmp_path / "figures"),
            "reports_dir": str(tmp_path / "reports"),
        },
        campaign={
            "max_authors_per_paper": 5,
            "max_papers_per_author": 10,
        },
    )


@pytest.fixture
def seeded_expansion(expander_config) -> tuple[str, str]:
    """Create campaign with a flagged seed paper that has authors."""
    init_db(expander_config.storage.database_url)
    init_async_db(expander_config.storage.database_url)

    campaign_id = "test-campaign-expand-001"
    paper_id = "test-paper-expand-001"

    with get_session() as session:
        campaign = Campaign(
            id=campaign_id,
            name="Expansion Test",
            mode="network_expansion",
            status="expanding",
            max_papers=50,
            llm_budget=10,
            papers_discovered=1,
        )
        session.add(campaign)

        paper = Paper(
            id=paper_id,
            title="Flagged Paper",
            doi="10.1234/flagged.001",
            authors_json=json.dumps(["Alice Smith", "Bob Jones", "Charlie Brown"]),
            source="test",
            status="complete",
        )
        session.add(paper)

        cp = CampaignPaper(
            campaign_id=campaign_id,
            paper_id=paper_id,
            source="seed",
            depth=0,
            triage_status="complete",
            final_risk="high",
            auto_risk_score=60.0,
        )
        session.add(cp)

    return campaign_id, paper_id


def _make_mock_work(doi: str, title: str, authors: list[str]) -> MagicMock:
    """Create a mock OpenAlex work result."""
    work = MagicMock()
    work.doi = doi
    work.title = title
    work.journal = "Test Journal"
    work.citation_count = 100
    work.publication_year = 2023
    work.authors = []
    for name in authors:
        author = MagicMock()
        author.name = name
        author.openalex_id = f"A{hash(name) % 10000}"
        work.authors.append(author)
    return work


def _make_mock_retraction_result(total: int = 0) -> MagicMock:
    result = MagicMock()
    result.total_retractions = total
    result.total_expressions_of_concern = 0
    result.retracted_dois = []
    return result


class TestExpandFromPaper:
    @pytest.mark.asyncio
    async def test_expands_from_seed_paper(self, expander_config, seeded_expansion):
        campaign_id, paper_id = seeded_expansion

        # Mock OpenAlex to return 2 new papers for Alice Smith
        mock_works = [
            _make_mock_work("10.1234/new.001", "New Paper 1", ["Alice Smith", "Dave Wilson"]),
            _make_mock_work("10.1234/new.002", "New Paper 2", ["Alice Smith", "Eve Adams"]),
        ]

        with patch(
            "rosette.discovery.openalex.search_works",
            new_callable=AsyncMock,
            return_value=mock_works,
        ):
            with patch(
                "rosette.discovery.retraction_watch.check_author_retractions",
                new_callable=AsyncMock,
                return_value=_make_mock_retraction_result(),
            ):
                expander = NetworkExpander(expander_config, campaign_id)
                new_ids = await expander.expand_from_paper(paper_id, depth=1)

        # Should have added papers (exact count depends on author iteration)
        assert len(new_ids) > 0

        # Verify papers were created in DB
        with get_session() as session:
            from sqlalchemy import select

            cps = (
                session.execute(
                    select(CampaignPaper)
                    .where(CampaignPaper.campaign_id == campaign_id)
                    .where(CampaignPaper.depth == 1)
                )
                .scalars()
                .all()
            )
            assert len(cps) > 0
            for cp in cps:
                assert cp.source == "network_expansion"
                assert cp.triage_status == "pending"

    @pytest.mark.asyncio
    async def test_respects_max_papers_cap(self, expander_config, seeded_expansion):
        campaign_id, paper_id = seeded_expansion

        # Set campaign max_papers to current discovered count
        with get_session() as session:
            campaign = session.get(Campaign, campaign_id)
            campaign.max_papers = 1  # Already have 1 paper
            campaign.papers_discovered = 1

        mock_works = [
            _make_mock_work("10.1234/capped.001", "Capped Paper", ["Alice Smith"]),
        ]

        with patch(
            "rosette.discovery.openalex.search_works",
            new_callable=AsyncMock,
            return_value=mock_works,
        ):
            with patch(
                "rosette.discovery.retraction_watch.check_author_retractions",
                new_callable=AsyncMock,
                return_value=_make_mock_retraction_result(),
            ):
                expander = NetworkExpander(expander_config, campaign_id)
                new_ids = await expander.expand_from_paper(paper_id, depth=1)

        assert len(new_ids) == 0

    @pytest.mark.asyncio
    async def test_skips_duplicate_dois(self, expander_config, seeded_expansion):
        campaign_id, paper_id = seeded_expansion

        # Return a paper with the same DOI as our seed
        mock_works = [
            _make_mock_work("10.1234/flagged.001", "Same Paper", ["Alice Smith"]),
        ]

        with patch(
            "rosette.discovery.openalex.search_works",
            new_callable=AsyncMock,
            return_value=mock_works,
        ):
            with patch(
                "rosette.discovery.retraction_watch.check_author_retractions",
                new_callable=AsyncMock,
                return_value=_make_mock_retraction_result(),
            ):
                expander = NetworkExpander(expander_config, campaign_id)
                new_ids = await expander.expand_from_paper(paper_id, depth=1)

        # Seed paper already in campaign, should not be added again
        assert "test-paper-expand-001" not in new_ids

    @pytest.mark.asyncio
    async def test_no_authors_returns_empty(self, expander_config, seeded_expansion):
        campaign_id, _ = seeded_expansion

        # Create a paper with no authors
        no_authors_id = "test-paper-no-authors"
        with get_session() as session:
            session.add(
                Paper(
                    id=no_authors_id,
                    title="No Authors",
                    source="test",
                    status="pending",
                    authors_json=None,
                )
            )
            session.add(
                CampaignPaper(
                    campaign_id=campaign_id,
                    paper_id=no_authors_id,
                    source="seed",
                    depth=0,
                    triage_status="pending",
                )
            )

        expander = NetworkExpander(expander_config, campaign_id)
        new_ids = await expander.expand_from_paper(no_authors_id, depth=1)
        assert new_ids == []


class TestExpandDepth:
    @pytest.mark.asyncio
    async def test_expand_depth_targets_flagged_papers(self, expander_config, seeded_expansion):
        campaign_id, paper_id = seeded_expansion

        mock_works = [
            _make_mock_work("10.1234/depth.001", "Depth 1 Paper", ["Alice Smith"]),
        ]

        with patch(
            "rosette.discovery.openalex.search_works",
            new_callable=AsyncMock,
            return_value=mock_works,
        ):
            with patch(
                "rosette.discovery.retraction_watch.check_author_retractions",
                new_callable=AsyncMock,
                return_value=_make_mock_retraction_result(),
            ):
                expander = NetworkExpander(expander_config, campaign_id)
                added = await expander.expand_depth(target_depth=1)

        # Should expand from the flagged seed paper (final_risk="high")
        assert added > 0

    @pytest.mark.asyncio
    async def test_expand_depth_ignores_clean_papers(self, expander_config, seeded_expansion):
        campaign_id, paper_id = seeded_expansion

        # Change seed paper to clean risk
        with get_session() as session:
            from sqlalchemy import select

            cp = (
                session.execute(
                    select(CampaignPaper)
                    .where(CampaignPaper.campaign_id == campaign_id)
                    .where(CampaignPaper.paper_id == paper_id)
                )
                .scalars()
                .first()
            )
            cp.final_risk = "clean"

        expander = NetworkExpander(expander_config, campaign_id)
        added = await expander.expand_depth(target_depth=1)
        assert added == 0


class TestCheckAuthorHistory:
    @pytest.mark.asyncio
    async def test_updates_author_retraction_count(self, expander_config, seeded_expansion):
        campaign_id, _ = seeded_expansion

        # Create an author record
        author_id = "test-author-001"
        with get_session() as session:
            session.add(Author(id=author_id, name="Alice Smith"))

        retraction_result = _make_mock_retraction_result(total=3)

        with patch(
            "rosette.discovery.retraction_watch.check_author_retractions",
            new_callable=AsyncMock,
            return_value=retraction_result,
        ):
            expander = NetworkExpander(expander_config, campaign_id)
            history = await expander.check_author_history("Alice Smith", author_id)

        assert history["total_retractions"] == 3

        with get_session() as session:
            author = session.get(Author, author_id)
            assert author.retraction_count == 3


class TestEnsureAuthorRecord:
    @pytest.mark.asyncio
    async def test_creates_new_author(self, expander_config, seeded_expansion):
        campaign_id, _ = seeded_expansion

        expander = NetworkExpander(expander_config, campaign_id)
        author_id = await expander._ensure_author_record("New Author", "New Author")

        assert author_id is not None
        with get_session() as session:
            author = session.get(Author, author_id)
            assert author is not None
            assert author.name == "New Author"

    @pytest.mark.asyncio
    async def test_returns_existing_author(self, expander_config, seeded_expansion):
        campaign_id, _ = seeded_expansion

        existing_id = "existing-author-001"
        with get_session() as session:
            session.add(Author(id=existing_id, name="Existing Author"))

        expander = NetworkExpander(expander_config, campaign_id)
        author_id = await expander._ensure_author_record("Existing Author", "Existing Author")
        assert author_id == existing_id

    @pytest.mark.asyncio
    async def test_extracts_orcid_from_dict(self, expander_config, seeded_expansion):
        campaign_id, _ = seeded_expansion

        author_info = {"name": "Dict Author", "orcid": "0000-0001-2345-6789"}

        expander = NetworkExpander(expander_config, campaign_id)
        author_id = await expander._ensure_author_record("Dict Author", author_info)

        with get_session() as session:
            author = session.get(Author, author_id)
            assert author.orcid == "0000-0001-2345-6789"
