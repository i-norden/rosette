"""Tests for shared run_analysis wrapper functions."""

from __future__ import annotations


from snoopy.analysis.run_analysis import (
    run_image_forensics,
    run_intra_paper_cross_ref,
    run_western_blot_analysis,
)
from snoopy.analysis.cross_reference import compute_phash


class TestRunImageForensics:
    def test_run_image_forensics_returns_findings(self, sample_image: str) -> None:
        """Basic forensics on a sample image returns a list (possibly empty)."""
        findings = run_image_forensics(sample_image)
        assert isinstance(findings, list)
        # Each finding should be a dict with required keys
        for f in findings:
            assert isinstance(f, dict)
            assert "title" in f
            assert "analysis_type" in f
            assert "severity" in f
            assert "confidence" in f

    def test_run_image_forensics_clean_image(self, clean_image: str) -> None:
        """A clean gradient image should produce few or no high-severity findings."""
        findings = run_image_forensics(clean_image)
        assert isinstance(findings, list)
        high_findings = [f for f in findings if f["severity"] in ("high", "critical")]
        # A simple gradient image should not trigger high-severity alarms
        # Note: some minor findings may occur due to noise analysis thresholds
        assert len(high_findings) == 0


class TestRunIntraPaperCrossRef:
    def test_run_intra_paper_cross_ref_no_duplicates(self, sample_image: str, clean_image: str) -> None:
        """Distinct images within the same paper should find no duplicates."""
        hash1 = compute_phash(sample_image)
        hash2 = compute_phash(clean_image)

        figure_results = [
            {"image": "fig1.png", "phash": hash1, "ahash": None},
            {"image": "fig2.png", "phash": hash2, "ahash": None},
        ]

        findings = run_intra_paper_cross_ref(figure_results)
        assert isinstance(findings, list)
        # Very different images should not produce hash matches
        assert len(findings) == 0

    def test_run_intra_paper_cross_ref_duplicate_detected(self, sample_image: str) -> None:
        """Identical images should be detected as intra-paper duplicates."""
        hash1 = compute_phash(sample_image)

        figure_results = [
            {"image": "fig1.png", "phash": hash1, "ahash": None},
            {"image": "fig2.png", "phash": hash1, "ahash": None},
        ]

        findings = run_intra_paper_cross_ref(figure_results)
        assert isinstance(findings, list)
        assert len(findings) >= 1
        assert findings[0]["analysis_type"] == "phash"


class TestRunWesternBlotAnalysis:
    def test_run_western_blot_analysis(self, sample_image: str) -> None:
        """Western blot analysis on a sample image returns a list."""
        findings = run_western_blot_analysis(sample_image)
        assert isinstance(findings, list)
        # Each finding should be a dict with required keys
        for f in findings:
            assert isinstance(f, dict)
            assert "title" in f
            assert "analysis_type" in f
            assert f["analysis_type"] == "western_blot"
