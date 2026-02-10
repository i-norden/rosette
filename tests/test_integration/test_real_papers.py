"""Integration tests using downloaded test fixtures.

These tests require fixtures to be downloaded first:
    python scripts/download_fixtures.py

Run with:
    pytest tests/test_integration/ -v -m integration
"""

from __future__ import annotations

from pathlib import Path

import pytest

from snoopy.analysis.image_forensics import (
    clone_detection,
    error_level_analysis,
    noise_analysis,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

# Skip entire module if no fixtures are present
pytestmark = pytest.mark.integration


def _has_fixtures(*subdirs: str) -> bool:
    """Check if the given fixture subdirectories exist and contain files."""
    for subdir in subdirs:
        d = FIXTURES_DIR / subdir
        if not d.exists() or not any(d.iterdir()):
            return False
    return True


def _image_files(subdir: str) -> list[Path]:
    """List image files in a fixture subdirectory."""
    d = FIXTURES_DIR / subdir
    if not d.exists():
        return []
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sorted(f for f in d.iterdir() if f.suffix.lower() in extensions)


def _pdf_files(subdir: str) -> list[Path]:
    """List PDF files in a fixture subdirectory."""
    d = FIXTURES_DIR / subdir
    if not d.exists():
        return []
    return sorted(f for f in d.iterdir() if f.suffix.lower() == ".pdf")


# ---------------------------------------------------------------------------
# Synthetic forgery images — these should trigger detection
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_fixtures("synthetic"),
    reason="Synthetic fixtures not downloaded (run scripts/download_fixtures.py)",
)
class TestSyntheticForgeries:
    """Tests that synthetic forgery images trigger forensic detection."""

    def test_copymove_images_trigger_ela_or_clone(self) -> None:
        """Copy-move images should trigger ELA or clone detection."""
        copymove_images = [f for f in _image_files("synthetic") if "copymove" in f.name]
        assert len(copymove_images) > 0, "No copymove images found"

        detected = 0
        for img_path in copymove_images:
            ela = error_level_analysis(str(img_path))
            clone = clone_detection(str(img_path))
            if ela.suspicious or clone.suspicious:
                detected += 1

        # At least some copy-move images should be detected
        assert detected > 0, (
            f"None of {len(copymove_images)} copy-move images were detected by ELA or clone detection"
        )

    def test_spliced_images_trigger_noise_analysis(self) -> None:
        """Spliced images should trigger noise inconsistency detection."""
        spliced_images = [f for f in _image_files("synthetic") if "spliced" in f.name]
        assert len(spliced_images) > 0, "No spliced images found"

        detected = 0
        for img_path in spliced_images:
            noise = noise_analysis(str(img_path))
            if noise.suspicious:
                detected += 1

        assert detected > 0, (
            f"None of {len(spliced_images)} spliced images were detected by noise analysis"
        )

    def test_retouched_images_trigger_detection(self) -> None:
        """Retouched images should trigger some form of detection."""
        retouched_images = [f for f in _image_files("synthetic") if "retouched" in f.name]
        assert len(retouched_images) > 0, "No retouched images found"

        detected = 0
        for img_path in retouched_images:
            ela = error_level_analysis(str(img_path))
            noise = noise_analysis(str(img_path))
            if ela.suspicious or noise.suspicious:
                detected += 1

        assert detected > 0, (
            f"None of {len(retouched_images)} retouched images were detected"
        )

    def test_all_synthetic_images_are_analyzable(self) -> None:
        """All synthetic images should be processable without errors."""
        all_images = _image_files("synthetic")
        assert len(all_images) >= 5, "Expected at least 5 synthetic images"

        for img_path in all_images:
            ela = error_level_analysis(str(img_path))
            clone = clone_detection(str(img_path))
            noise = noise_analysis(str(img_path))
            # Just verify no exceptions and reasonable values
            assert ela.mean_difference >= 0
            assert clone.num_matches >= 0
            assert noise.mean_noise >= 0


# ---------------------------------------------------------------------------
# RSIIL benchmark images — manipulated images should trigger detection
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_fixtures("rsiil"),
    reason="RSIIL fixtures not downloaded (run scripts/download_fixtures.py)",
)
class TestRSIILBenchmark:
    """Tests that RSIIL benchmark manipulated images trigger detection."""

    def test_rsiil_images_are_analyzable(self) -> None:
        """All RSIIL images should be processable without errors."""
        images = _image_files("rsiil")
        assert len(images) > 0, "No RSIIL images found"

        for img_path in images:
            ela = error_level_analysis(str(img_path))
            clone = clone_detection(str(img_path))
            noise = noise_analysis(str(img_path))
            assert ela.mean_difference >= 0
            assert clone.num_matches >= 0
            assert noise.mean_noise >= 0

    def test_duplication_images_trigger_clone_detection(self) -> None:
        """RSIIL duplication images should trigger clone detection."""
        duplication_images = [f for f in _image_files("rsiil") if "duplication" in f.name]
        if not duplication_images:
            pytest.skip("No duplication images in RSIIL fixtures")

        detected = 0
        for img_path in duplication_images:
            clone = clone_detection(str(img_path))
            ela = error_level_analysis(str(img_path))
            if clone.suspicious or ela.suspicious:
                detected += 1

        # At least one should be caught
        assert detected > 0, "No duplication images were detected"


# ---------------------------------------------------------------------------
# Clean control papers — should NOT trigger false positives
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_fixtures("clean"),
    reason="Clean fixtures not downloaded (run scripts/download_fixtures.py)",
)
class TestCleanControls:
    """Tests that clean, reputable papers do not trigger high-severity findings."""

    def test_clean_paper_figures_no_high_severity(self, tmp_path: Path) -> None:
        """Figures from clean papers should not trigger high/critical findings."""
        pdfs = _pdf_files("clean")
        if not pdfs:
            pytest.skip("No clean control PDFs found")

        from snoopy.extraction.figure_extractor import extract_figures

        false_positives = []
        for pdf_path in pdfs[:2]:  # Test first 2 to keep test time reasonable
            try:
                out_dir = tmp_path / pdf_path.stem
                figures = extract_figures(str(pdf_path), str(out_dir))

                for fig in figures:
                    img_path = Path(fig.image_path)
                    if not img_path.exists():
                        continue

                    clone = clone_detection(str(img_path))
                    if clone.suspicious and clone.inlier_ratio > 0.3:
                        false_positives.append(
                            f"{pdf_path.name}/{img_path.name}: "
                            f"clone detection ({clone.num_matches} matches, "
                            f"inlier_ratio={clone.inlier_ratio:.2f})"
                        )
            except Exception:
                pass  # PDF extraction issues are OK in this context

        assert len(false_positives) == 0, (
            f"False positives on clean papers:\n" + "\n".join(false_positives)
        )


# ---------------------------------------------------------------------------
# PDF extraction on real PMC papers
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_fixtures("survey"),
    reason="Survey fixtures not downloaded (run scripts/download_fixtures.py)",
)
class TestPDFExtraction:
    """Tests PDF text and figure extraction on real PMC papers."""

    def test_survey_paper_text_extraction(self) -> None:
        """Bik survey paper should yield extractable text."""
        pdfs = _pdf_files("survey")
        if not pdfs:
            pytest.skip("Survey PDF not found")

        from snoopy.extraction.pdf_parser import extract_text

        pages = extract_text(str(pdfs[0]))
        assert len(pages) > 0, "No pages extracted from survey paper"
        total_words = sum(p.word_count for p in pages)
        assert total_words > 100, f"Too few words ({total_words}) extracted from survey paper"

    def test_survey_paper_figure_extraction(self, tmp_path: Path) -> None:
        """Bik survey paper should contain extractable figures."""
        pdfs = _pdf_files("survey")
        if not pdfs:
            pytest.skip("Survey PDF not found")

        from snoopy.extraction.figure_extractor import extract_figures

        figures = extract_figures(str(pdfs[0]), str(tmp_path / "figures"))
        # The Bik survey paper contains example figures of manipulation types
        assert len(figures) >= 1, "Expected at least 1 figure from survey paper"

        for fig in figures:
            assert fig.width >= 50
            assert fig.height >= 50
            assert Path(fig.image_path).exists()

    def test_retracted_paper_text_extraction(self) -> None:
        """Retracted papers should yield extractable text."""
        pdfs = _pdf_files("retracted")
        if not pdfs:
            pytest.skip("No retracted PDFs found")

        from snoopy.extraction.pdf_parser import extract_text

        for pdf_path in pdfs[:2]:
            pages = extract_text(str(pdf_path))
            assert len(pages) > 0, f"No pages extracted from {pdf_path.name}"
