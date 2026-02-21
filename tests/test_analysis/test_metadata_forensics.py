"""Tests for EXIF/ICC metadata forensics analysis."""

from __future__ import annotations

from rosette.analysis.metadata_forensics import MetadataForensicsResult, analyze_metadata


class TestAnalyzeMetadataClean:
    def test_analyze_metadata_clean(self, clean_image: str) -> None:
        """A clean image with no suspicious metadata produces no high-confidence findings."""
        result = analyze_metadata(clean_image)
        assert isinstance(result, MetadataForensicsResult)
        # A programmatically generated JPEG should not have editing software signatures
        # It may or may not have stripped-metadata findings, but nothing high-confidence
        high_confidence = [f for f in result.findings if f.confidence > 0.5]
        assert len(high_confidence) == 0


class TestAnalyzeMetadataNonexistent:
    def test_analyze_metadata_nonexistent(self) -> None:
        """Handles missing file gracefully without raising an unhandled exception."""
        # analyze_metadata calls _extract_exif which catches exceptions
        # and returns empty data, so it should not crash
        result = analyze_metadata("/nonexistent/path/image.png")
        assert isinstance(result, MetadataForensicsResult)
        # No findings should be produced for a nonexistent file
        # (internal functions catch exceptions and return empty results)
        assert result.suspicious is False
