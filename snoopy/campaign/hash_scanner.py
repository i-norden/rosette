"""Campaign-scale cross-paper image hash matching with prefix-bucketed index."""

from __future__ import annotations

import logging
from collections import defaultdict

from sqlalchemy import select

from snoopy.analysis.cross_reference import hash_distance
from snoopy.config import SnoopyConfig
from snoopy.db.models import Campaign, CampaignPaper, Figure, ImageHashMatch
from snoopy.db.session import get_async_session, get_session

logger = logging.getLogger(__name__)

# Prefix length for hash bucketing (first N hex chars)
_PREFIX_LEN = 4
_PAGE_SIZE = 500


class CampaignHashScanner:
    """Prefix-bucketed hash scanner for efficient cross-paper image matching."""

    def __init__(self, config: SnoopyConfig, campaign_id: str | None = None):
        self.config = config
        self.campaign_id = campaign_id
        self.max_distance = config.campaign.hash_match_max_distance
        # In-memory index: prefix -> [(figure_id, paper_id, full_hash)]
        self._index: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    def _build_index(self, paper_ids: list[str] | None = None) -> None:
        """Build prefix-bucketed hash index from DB.

        Args:
            paper_ids: If provided, only index figures from these papers.
                       Otherwise, index all figures in the database.
        """
        self._index.clear()

        with get_session() as session:
            offset = 0
            while True:
                query = select(Figure.id, Figure.paper_id, Figure.phash).where(
                    Figure.phash.isnot(None)
                )
                if paper_ids is not None:
                    query = query.where(Figure.paper_id.in_(paper_ids))

                rows = session.execute(query.offset(offset).limit(_PAGE_SIZE)).all()

                if not rows:
                    break

                for row in rows:
                    phash = row.phash
                    if phash and len(phash) >= _PREFIX_LEN:
                        prefix = phash[:_PREFIX_LEN]
                        self._index[prefix].append((row.id, row.paper_id, phash))

                offset += _PAGE_SIZE

        total = sum(len(v) for v in self._index.values())
        logger.info("Hash index built: %d figures in %d buckets", total, len(self._index))

    async def scan_all_pairs(self) -> list[ImageHashMatch]:
        """Full database scan for cross-paper hash matches."""
        self._build_index()
        return await self._find_matches_in_index()

    async def scan_incremental(self, new_paper_ids: list[str]) -> list[ImageHashMatch]:
        """Compare new papers' figures against existing index.

        Builds index from all papers NOT in new_paper_ids, then compares
        new papers' figures against the existing index.
        """
        # Build index from all existing figures
        self._build_index()

        # Get new figures
        new_figures: list[tuple[str, str, str]] = []
        with get_session() as session:
            for paper_id in new_paper_ids:
                rows = session.execute(
                    select(Figure.id, Figure.paper_id, Figure.phash)
                    .where(Figure.paper_id == paper_id)
                    .where(Figure.phash.isnot(None))
                ).all()
                for row in rows:
                    if row.phash and len(row.phash) >= _PREFIX_LEN:
                        new_figures.append((row.id, row.paper_id, row.phash))

        # Compare new figures against index
        matches: list[ImageHashMatch] = []
        seen_pairs: set[tuple[str, str]] = set()

        for fig_id, paper_id, phash in new_figures:
            prefix = phash[:_PREFIX_LEN]
            # Check same bucket and neighboring buckets
            for bucket_prefix in self._nearby_prefixes(prefix):
                for other_id, other_paper_id, other_hash in self._index.get(bucket_prefix, []):
                    # Skip same paper
                    if paper_id == other_paper_id:
                        continue
                    # Skip already seen pairs
                    pair_key = (min(fig_id, other_id), max(fig_id, other_id))
                    if pair_key in seen_pairs:
                        continue

                    try:
                        dist = hash_distance(phash, other_hash)
                    except ValueError:
                        continue

                    if dist <= self.max_distance:
                        seen_pairs.add(pair_key)
                        match = ImageHashMatch(
                            campaign_id=self.campaign_id,
                            figure_id_a=fig_id,
                            figure_id_b=other_id,
                            paper_id_a=paper_id,
                            paper_id_b=other_paper_id,
                            hash_type="phash",
                            hash_distance=dist,
                        )
                        matches.append(match)

        # Persist matches
        if matches:
            await self._persist_matches(matches)

        logger.info(
            "Incremental scan: %d new figures, %d matches found",
            len(new_figures),
            len(matches),
        )
        return matches

    async def _find_matches_in_index(self) -> list[ImageHashMatch]:
        """Find all cross-paper matches within the bucketed index."""
        matches: list[ImageHashMatch] = []
        seen_pairs: set[tuple[str, str]] = set()

        for prefix, entries in self._index.items():
            # Compare within bucket
            for i, (fig_a, paper_a, hash_a) in enumerate(entries):
                for fig_b, paper_b, hash_b in entries[i + 1 :]:
                    if paper_a == paper_b:
                        continue
                    pair_key = (min(fig_a, fig_b), max(fig_a, fig_b))
                    if pair_key in seen_pairs:
                        continue

                    try:
                        dist = hash_distance(hash_a, hash_b)
                    except ValueError:
                        continue

                    if dist <= self.max_distance:
                        seen_pairs.add(pair_key)
                        match = ImageHashMatch(
                            campaign_id=self.campaign_id,
                            figure_id_a=fig_a,
                            figure_id_b=fig_b,
                            paper_id_a=paper_a,
                            paper_id_b=paper_b,
                            hash_type="phash",
                            hash_distance=dist,
                        )
                        matches.append(match)

            # Also compare with nearby prefixes for cross-bucket matches
            for neighbor in self._nearby_prefixes(prefix):
                if neighbor <= prefix:  # avoid double-counting
                    continue
                for fig_a, paper_a, hash_a in entries:
                    for fig_b, paper_b, hash_b in self._index.get(neighbor, []):
                        if paper_a == paper_b:
                            continue
                        pair_key = (min(fig_a, fig_b), max(fig_a, fig_b))
                        if pair_key in seen_pairs:
                            continue

                        try:
                            dist = hash_distance(hash_a, hash_b)
                        except ValueError:
                            continue

                        if dist <= self.max_distance:
                            seen_pairs.add(pair_key)
                            match = ImageHashMatch(
                                campaign_id=self.campaign_id,
                                figure_id_a=fig_a,
                                figure_id_b=fig_b,
                                paper_id_a=paper_a,
                                paper_id_b=paper_b,
                                hash_type="phash",
                                hash_distance=dist,
                            )
                            matches.append(match)

        # Persist all matches
        if matches:
            await self._persist_matches(matches)

        logger.info("Full scan: %d cross-paper matches found", len(matches))
        return matches

    async def _persist_matches(self, matches: list[ImageHashMatch]) -> None:
        """Save hash matches to database, skipping duplicates."""
        async with get_async_session() as session:
            for match in matches:
                # Check for existing match
                existing = await session.execute(
                    select(ImageHashMatch).where(
                        ImageHashMatch.figure_id_a == match.figure_id_a,
                        ImageHashMatch.figure_id_b == match.figure_id_b,
                    )
                )
                if not existing.scalars().first():
                    session.add(match)

            # Update campaign flagged count if applicable
            if self.campaign_id:
                campaign = await session.get(Campaign, self.campaign_id)
                if campaign:
                    # Count unique papers with hash matches
                    paper_ids_with_matches: set[str] = set()
                    for m in matches:
                        paper_ids_with_matches.add(str(m.paper_id_a))
                        paper_ids_with_matches.add(str(m.paper_id_b))

                    # Update flagged count for campaign papers
                    for pid in paper_ids_with_matches:
                        cp_result = await session.execute(
                            select(CampaignPaper)
                            .where(CampaignPaper.campaign_id == self.campaign_id)
                            .where(CampaignPaper.paper_id == pid)
                        )
                        cp = cp_result.scalars().first()
                        if cp and not cp.llm_promoted:
                            # Hash match may trigger re-scoring
                            pass

    def _nearby_prefixes(self, prefix: str) -> list[str]:
        """Get hash prefixes within Hamming distance 1 of the given prefix.

        For hex prefixes, each character has 16 possible values. We generate
        all prefixes that differ in exactly one hex digit.
        """
        neighbors = [prefix]
        hex_chars = "0123456789abcdef"
        for i, char in enumerate(prefix):
            for h in hex_chars:
                if h != char:
                    neighbor = prefix[:i] + h + prefix[i + 1 :]
                    neighbors.append(neighbor)
        return neighbors
