"""Tests for priority scoring algorithm."""

from rosette.discovery.priority import (
    PaperMetadata,
    check_institution_in_top100,
    compute_priority,
)


class TestComputePriority:
    def test_zero_score_for_empty_metadata(self):
        paper = PaperMetadata(
            citation_count=0,
            journal_quartile=4,
            influential_citations=0,
            max_author_hindex=0,
            institution_in_top100=False,
            has_retraction_concern=False,
            year=2026,
            has_image_heavy_methods=False,
        )
        score = compute_priority(paper)
        assert 0 <= score <= 100
        # Low score expected: Q4 journal gives 0.25*20=5, non-top100 gives 0.3*10=3,
        # no image methods gives 0.5*5=2.5
        assert score < 15

    def test_high_score_for_high_impact(self):
        paper = PaperMetadata(
            citation_count=5000,
            journal_quartile=1,
            influential_citations=500,
            max_author_hindex=60,
            institution_in_top100=True,
            has_retraction_concern=True,
            year=2010,
            has_image_heavy_methods=True,
        )
        score = compute_priority(paper)
        assert score > 70

    def test_score_is_between_0_and_100(self):
        paper = PaperMetadata(
            citation_count=100,
            journal_quartile=2,
            influential_citations=50,
            max_author_hindex=30,
            institution_in_top100=False,
            has_retraction_concern=False,
            year=2020,
            has_image_heavy_methods=True,
        )
        score = compute_priority(paper)
        assert 0 <= score <= 100

    def test_citation_cap(self):
        """Score should not increase beyond 10K citations."""
        paper_10k = PaperMetadata(citation_count=10_000)
        paper_100k = PaperMetadata(citation_count=100_000)
        # Both should get the same citation component
        s1 = compute_priority(paper_10k)
        s2 = compute_priority(paper_100k)
        assert s1 == s2

    def test_retraction_concern_boosts_score(self):
        base = PaperMetadata(citation_count=100, journal_quartile=2)
        with_concern = PaperMetadata(
            citation_count=100, journal_quartile=2, has_retraction_concern=True
        )
        assert compute_priority(with_concern) > compute_priority(base)

    def test_q1_higher_than_q4(self):
        q1 = PaperMetadata(journal_quartile=1)
        q4 = PaperMetadata(journal_quartile=4)
        assert compute_priority(q1) > compute_priority(q4)


class TestInstitutionCheck:
    def test_exact_match(self):
        assert check_institution_in_top100(["Harvard University"])

    def test_substring_match(self):
        # "Harvard University" is a substring of "affiliated with Harvard University"
        assert check_institution_in_top100(["affiliated with Harvard University"])

    def test_case_insensitive(self):
        assert check_institution_in_top100(["HARVARD UNIVERSITY"])

    def test_no_match(self):
        assert not check_institution_in_top100(["University of Nowhere"])

    def test_empty_list(self):
        assert not check_institution_in_top100([])
