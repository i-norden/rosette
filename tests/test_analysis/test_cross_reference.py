"""Tests for cross-reference image hashing and duplicate detection."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from snoopy.analysis.cross_reference import (
    build_hash_index,
    compute_phash,
    find_cross_paper_duplicates,
    hash_distance,
)
from snoopy.db.models import Base, Figure, Paper
from snoopy.db.session import get_session, init_db


class TestComputePhash:
    def test_compute_phash_consistent(self, sample_image: str) -> None:
        """Same image returns the same perceptual hash on repeated calls."""
        hash1 = compute_phash(sample_image)
        hash2 = compute_phash(sample_image)
        assert hash1 is not None
        assert hash2 is not None
        assert hash1 == hash2

    def test_compute_phash_different_images(self, sample_image: str, clean_image: str) -> None:
        """Different images return different perceptual hashes."""
        hash1 = compute_phash(sample_image)
        hash2 = compute_phash(clean_image)
        assert hash1 is not None
        assert hash2 is not None
        assert hash1 != hash2


class TestHashDistance:
    def test_hash_distance_identical(self, sample_image: str) -> None:
        """Distance is 0 for the same hash compared to itself."""
        h = compute_phash(sample_image)
        assert h is not None
        assert hash_distance(h, h) == 0

    def test_hash_distance_different(self, sample_image: str, clean_image: str) -> None:
        """Distance is non-zero for hashes of different images."""
        h1 = compute_phash(sample_image)
        h2 = compute_phash(clean_image)
        assert h1 is not None
        assert h2 is not None
        dist = hash_distance(h1, h2)
        assert dist > 0

    def test_hash_distance_mismatched_length_raises(self) -> None:
        """ValueError is raised when hash strings have different lengths."""
        short_hash = "abcd1234"
        long_hash = "abcd12345678abcd"
        with pytest.raises(ValueError, match="Hash length mismatch"):
            hash_distance(short_hash, long_hash)


class TestFindCrossPaperDuplicates:
    def test_find_cross_paper_duplicates_no_matches(self, test_config, tmp_path) -> None:
        """No matches returned when papers have completely distinct figures."""
        init_db(test_config.storage.database_url)

        # Create two distinct images with very different content
        img_a = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        path_a = str(tmp_path / "fig_a.png")
        img_a.save(path_a)

        img_b = Image.fromarray(
            np.full((100, 100, 3), 255, dtype=np.uint8)
        )
        path_b = str(tmp_path / "fig_b.png")
        img_b.save(path_b)

        hash_a = compute_phash(path_a)
        hash_b = compute_phash(path_b)

        with get_session() as session:
            paper_a = Paper(id="paper-a", title="Paper A", status="complete")
            paper_b = Paper(id="paper-b", title="Paper B", status="complete")
            session.add(paper_a)
            session.add(paper_b)
            session.flush()

            fig_a = Figure(
                id="fig-a",
                paper_id="paper-a",
                image_path=path_a,
                phash=hash_a,
            )
            fig_b = Figure(
                id="fig-b",
                paper_id="paper-b",
                image_path=path_b,
                phash=hash_b,
            )
            session.add(fig_a)
            session.add(fig_b)

        result = find_cross_paper_duplicates("paper-a", max_distance=5)
        assert result.cross_paper_duplicates == 0
        assert len(result.matches) == 0


class TestBuildHashIndex:
    def test_build_hash_index_reads_stored_hashes(self, test_config, tmp_path) -> None:
        """Index is built from stored DB hashes, not recomputed from disk."""
        init_db(test_config.storage.database_url)

        with get_session() as session:
            paper = Paper(id="paper-idx", title="Index Paper", status="complete")
            session.add(paper)
            session.flush()

            fig1 = Figure(
                id="fig-idx-1",
                paper_id="paper-idx",
                image_path="/nonexistent/path1.png",
                phash="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            )
            fig2 = Figure(
                id="fig-idx-2",
                paper_id="paper-idx",
                image_path="/nonexistent/path2.png",
                phash="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            )
            session.add(fig1)
            session.add(fig2)

        index = build_hash_index()
        # The index should contain both hashes
        assert len(index) == 2
        hash1 = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        hash2 = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        assert hash1 in index
        assert hash2 in index
        assert ("fig-idx-1", "paper-idx") in index[hash1]
        assert ("fig-idx-2", "paper-idx") in index[hash2]
