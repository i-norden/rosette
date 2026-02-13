"""Tests for snoopy.campaign.hash_scanner module.

Tests prefix-bucketed hash matching, incremental scanning,
and match persistence.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from snoopy.campaign.hash_scanner import CampaignHashScanner, _PREFIX_LEN
from snoopy.config import SnoopyConfig
from snoopy.db.models import Campaign, CampaignPaper, Figure, ImageHashMatch, Paper
from snoopy.db.session import get_session, init_async_db, init_db


@pytest.fixture
def scanner_config(tmp_path) -> SnoopyConfig:
    db_path = tmp_path / "scanner_test.db"
    return SnoopyConfig(
        storage={
            "database_url": f"sqlite:///{db_path}",
            "pdf_dir": str(tmp_path / "pdfs"),
            "figures_dir": str(tmp_path / "figures"),
            "reports_dir": str(tmp_path / "reports"),
        },
        campaign={"hash_match_max_distance": 10},
    )


@pytest.fixture
def scanner_db(scanner_config) -> str:
    """Initialize DB, create campaign and papers with figures. Return campaign_id."""
    init_db(scanner_config.storage.database_url)
    init_async_db(scanner_config.storage.database_url)

    campaign_id = "test-campaign-scanner-001"

    with get_session() as session:
        campaign = Campaign(
            id=campaign_id,
            name="Scanner Test",
            mode="domain_scan",
            status="auto_analyzing",
            max_papers=100,
            llm_budget=10,
        )
        session.add(campaign)

        # Paper A with 2 figures
        paper_a = Paper(
            id="paper-a", title="Paper A", doi="10.1234/a",
            source="test", status="complete",
        )
        session.add(paper_a)

        fig_a1 = Figure(
            id="fig-a1", paper_id="paper-a",
            phash="abcd1234abcd1234",  # 16 hex chars
        )
        fig_a2 = Figure(
            id="fig-a2", paper_id="paper-a",
            phash="1111222233334444",
        )
        session.add(fig_a1)
        session.add(fig_a2)

        # Paper B with a figure very similar to paper A's fig_a1
        paper_b = Paper(
            id="paper-b", title="Paper B", doi="10.1234/b",
            source="test", status="complete",
        )
        session.add(paper_b)

        # Only differs in last char: Hamming distance = 4 bits (hex f vs 4 = 1111 vs 0100)
        fig_b1 = Figure(
            id="fig-b1", paper_id="paper-b",
            phash="abcd1234abcd123f",
        )
        session.add(fig_b1)

        # Paper C with a completely different hash
        paper_c = Paper(
            id="paper-c", title="Paper C", doi="10.1234/c",
            source="test", status="complete",
        )
        session.add(paper_c)

        fig_c1 = Figure(
            id="fig-c1", paper_id="paper-c",
            phash="ffffeeeeddddcccc",
        )
        session.add(fig_c1)

        # Campaign paper links
        for pid in ["paper-a", "paper-b", "paper-c"]:
            session.add(CampaignPaper(
                campaign_id=campaign_id, paper_id=pid,
                source="domain_scan", depth=0, triage_status="auto_done",
            ))

    return campaign_id


class TestNearbyPrefixes:
    def test_includes_original(self, scanner_config):
        scanner = CampaignHashScanner(scanner_config)
        neighbors = scanner._nearby_prefixes("abcd")
        assert "abcd" in neighbors

    def test_generates_correct_count(self, scanner_config):
        scanner = CampaignHashScanner(scanner_config)
        prefix = "abcd"
        neighbors = scanner._nearby_prefixes(prefix)
        # Original + 15 variants per position * 4 positions = 1 + 60 = 61
        assert len(neighbors) == 1 + (15 * _PREFIX_LEN)

    def test_neighbors_differ_by_one_char(self, scanner_config):
        scanner = CampaignHashScanner(scanner_config)
        neighbors = scanner._nearby_prefixes("0000")
        for n in neighbors:
            assert len(n) == 4
            diffs = sum(1 for a, b in zip("0000", n) if a != b)
            assert diffs <= 1


class TestBuildIndex:
    def test_builds_index_from_all_papers(self, scanner_config, scanner_db):
        scanner = CampaignHashScanner(scanner_config, scanner_db)
        scanner._build_index()

        total = sum(len(v) for v in scanner._index.values())
        assert total == 4  # 4 figures total

    def test_builds_index_filtered_by_paper_ids(self, scanner_config, scanner_db):
        scanner = CampaignHashScanner(scanner_config, scanner_db)
        scanner._build_index(paper_ids=["paper-a"])

        total = sum(len(v) for v in scanner._index.values())
        assert total == 2  # Paper A has 2 figures

    def test_index_uses_prefix_buckets(self, scanner_config, scanner_db):
        scanner = CampaignHashScanner(scanner_config, scanner_db)
        scanner._build_index()

        # fig-a1 and fig-b1 both have prefix "abcd" -> same bucket
        assert "abcd" in scanner._index
        abcd_entries = scanner._index["abcd"]
        paper_ids = {entry[1] for entry in abcd_entries}
        assert "paper-a" in paper_ids
        assert "paper-b" in paper_ids

    def test_skips_null_phash(self, scanner_config, scanner_db):
        with get_session() as session:
            session.add(Figure(
                id="fig-null", paper_id="paper-a", phash=None,
            ))

        scanner = CampaignHashScanner(scanner_config, scanner_db)
        scanner._build_index()

        total = sum(len(v) for v in scanner._index.values())
        assert total == 4  # null phash figure not indexed

    def test_skips_short_phash(self, scanner_config, scanner_db):
        with get_session() as session:
            session.add(Figure(
                id="fig-short", paper_id="paper-a", phash="ab",
            ))

        scanner = CampaignHashScanner(scanner_config, scanner_db)
        scanner._build_index()

        total = sum(len(v) for v in scanner._index.values())
        assert total == 4  # short phash not indexed


class TestScanAllPairs:
    @pytest.mark.asyncio
    async def test_finds_cross_paper_matches(self, scanner_config, scanner_db):
        scanner = CampaignHashScanner(scanner_config, scanner_db)
        await scanner.scan_all_pairs()

        # fig-a1 (abcd1234abcd1234) and fig-b1 (abcd1234abcd123f) should match
        # Verify via DB query since returned ORM objects may be detached
        from sqlalchemy import select
        with get_session() as session:
            db_matches = session.execute(select(ImageHashMatch)).scalars().all()
            match_pairs = set()
            for m in db_matches:
                match_pairs.add((str(m.figure_id_a), str(m.figure_id_b)))
            has_match = (
                ("fig-a1", "fig-b1") in match_pairs
                or ("fig-b1", "fig-a1") in match_pairs
            )
            assert has_match

    @pytest.mark.asyncio
    async def test_no_same_paper_matches(self, scanner_config, scanner_db):
        scanner = CampaignHashScanner(scanner_config, scanner_db)
        await scanner.scan_all_pairs()

        from sqlalchemy import select
        with get_session() as session:
            db_matches = session.execute(select(ImageHashMatch)).scalars().all()
            for m in db_matches:
                assert str(m.paper_id_a) != str(m.paper_id_b)

    @pytest.mark.asyncio
    async def test_matches_persisted_to_db(self, scanner_config, scanner_db):
        scanner = CampaignHashScanner(scanner_config, scanner_db)
        await scanner.scan_all_pairs()

        from sqlalchemy import select
        with get_session() as session:
            db_matches = session.execute(select(ImageHashMatch)).scalars().all()
            # Should have at least the fig-a1/fig-b1 match
            assert len(db_matches) >= 1


class TestScanIncremental:
    @pytest.mark.asyncio
    async def test_incremental_finds_new_matches(self, scanner_config, scanner_db):
        # Add a new paper with a similar figure
        with get_session() as session:
            session.add(Paper(
                id="paper-d", title="Paper D", doi="10.1234/d",
                source="test", status="complete",
            ))
            # Very similar to fig-a1 (abcd1234abcd1234)
            session.add(Figure(
                id="fig-d1", paper_id="paper-d",
                phash="abcd1234abcd1230",  # differs by 1 hex digit at end
            ))

        scanner = CampaignHashScanner(scanner_config, scanner_db)
        matches = await scanner.scan_incremental(["paper-d"])

        # Should find match between fig-d1 and fig-a1 (and possibly fig-b1)
        assert len(matches) > 0

    @pytest.mark.asyncio
    async def test_incremental_empty_for_unrelated(self, scanner_config, scanner_db):
        # Add paper with a completely different hash
        with get_session() as session:
            session.add(Paper(
                id="paper-e", title="Paper E", doi="10.1234/e",
                source="test", status="complete",
            ))
            session.add(Figure(
                id="fig-e1", paper_id="paper-e",
                phash="0000000000000000",
            ))

        scanner = CampaignHashScanner(scanner_config, scanner_db)
        # Use very low max_distance to exclude far hashes
        scanner.max_distance = 2
        matches = await scanner.scan_incremental(["paper-e"])

        # Should not match anything with max_distance=2
        assert len(matches) == 0


class TestPersistMatches:
    @pytest.mark.asyncio
    async def test_skips_duplicate_matches(self, scanner_config, scanner_db):
        scanner = CampaignHashScanner(scanner_config, scanner_db)

        match = ImageHashMatch(
            campaign_id=scanner_db,
            figure_id_a="fig-a1",
            figure_id_b="fig-b1",
            paper_id_a="paper-a",
            paper_id_b="paper-b",
            hash_type="phash",
            hash_distance=3,
        )

        # Persist twice
        await scanner._persist_matches([match])

        match2 = ImageHashMatch(
            campaign_id=scanner_db,
            figure_id_a="fig-a1",
            figure_id_b="fig-b1",
            paper_id_a="paper-a",
            paper_id_b="paper-b",
            hash_type="phash",
            hash_distance=3,
        )
        await scanner._persist_matches([match2])

        from sqlalchemy import select
        with get_session() as session:
            all_matches = session.execute(
                select(ImageHashMatch)
                .where(ImageHashMatch.figure_id_a == "fig-a1")
                .where(ImageHashMatch.figure_id_b == "fig-b1")
            ).scalars().all()
            assert len(all_matches) == 1
