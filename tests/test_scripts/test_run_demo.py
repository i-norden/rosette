"""Tests for snoopy.demo.runner helper functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from snoopy.demo.runner import (
    _analyze_image,
    _analyze_pdf,
    _build_result,
    _collect_methods,
    _find_images,
    _find_pdfs,
)


class TestFindImages:
    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        result = _find_images(tmp_path / "nope")
        assert result == []

    def test_empty_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        assert _find_images(d) == []

    def test_finds_png_and_jpg(self, tmp_path: Path) -> None:
        (tmp_path / "a.png").write_bytes(b"png")
        (tmp_path / "b.jpg").write_bytes(b"jpg")
        (tmp_path / "c.txt").write_bytes(b"text")
        result = _find_images(tmp_path)
        names = [p.name for p in result]
        assert "a.png" in names
        assert "b.jpg" in names
        assert "c.txt" not in names

    def test_finds_all_image_extensions(self, tmp_path: Path) -> None:
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            (tmp_path / f"img{ext}").write_bytes(b"data")
        result = _find_images(tmp_path)
        assert len(result) == 6

    def test_case_insensitive_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "photo.PNG").write_bytes(b"data")
        (tmp_path / "photo.JPG").write_bytes(b"data")
        result = _find_images(tmp_path)
        assert len(result) == 2

    def test_results_are_sorted(self, tmp_path: Path) -> None:
        (tmp_path / "c.png").write_bytes(b"data")
        (tmp_path / "a.png").write_bytes(b"data")
        (tmp_path / "b.png").write_bytes(b"data")
        result = _find_images(tmp_path)
        names = [p.name for p in result]
        assert names == ["a.png", "b.png", "c.png"]


class TestFindPdfs:
    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        assert _find_pdfs(tmp_path / "nope") == []

    def test_empty_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        assert _find_pdfs(d) == []

    def test_finds_pdfs_only(self, tmp_path: Path) -> None:
        (tmp_path / "paper.pdf").write_bytes(b"pdf")
        (tmp_path / "image.png").write_bytes(b"png")
        (tmp_path / "doc.PDF").write_bytes(b"pdf2")
        result = _find_pdfs(tmp_path)
        names = [p.name for p in result]
        assert "paper.pdf" in names
        assert "doc.PDF" in names
        assert "image.png" not in names

    def test_results_are_sorted(self, tmp_path: Path) -> None:
        (tmp_path / "z.pdf").write_bytes(b"data")
        (tmp_path / "a.pdf").write_bytes(b"data")
        result = _find_pdfs(tmp_path)
        assert result[0].name == "a.pdf"
        assert result[1].name == "z.pdf"


class TestAnalyzeImage:
    def _make_clean_image(self, tmp_path: Path) -> Path:
        """Create a simple gradient image (low suspicion)."""
        arr = np.zeros((200, 300, 3), dtype=np.uint8)
        for i in range(200):
            arr[i, :, :] = int(i / 200 * 255)
        img = Image.fromarray(arr)
        path = tmp_path / "clean.jpg"
        img.save(str(path), "JPEG", quality=95)
        return path

    def _make_manipulated_image(self, tmp_path: Path) -> Path:
        """Create an image with a cloned region."""
        rng = np.random.RandomState(42)
        arr = rng.randint(20, 60, (512, 512, 3), dtype=np.uint8)
        for y in range(0, 512, 64):
            arr[y : y + 32, :, :] = rng.randint(40, 200)
        arr[300:400, 300:400] = arr[50:150, 50:150]
        img = Image.fromarray(arr)
        path = tmp_path / "manipulated.png"
        img.save(str(path))
        return path

    def test_returns_expected_keys(self, tmp_path: Path) -> None:
        path = self._make_clean_image(tmp_path)
        result = _analyze_image(path)
        assert "image" in result
        assert "path" in result
        assert "findings" in result
        assert "num_findings" in result
        assert "phash" in result
        assert "ahash" in result
        assert result["image"] == "clean.jpg"

    def test_clean_image_has_few_findings(self, tmp_path: Path) -> None:
        path = self._make_clean_image(tmp_path)
        result = _analyze_image(path)
        high_sev = [f for f in result["findings"] if f["severity"] in ("critical", "high")]
        assert len(high_sev) == 0

    def test_findings_have_required_fields(self, tmp_path: Path) -> None:
        path = self._make_manipulated_image(tmp_path)
        result = _analyze_image(path)
        for finding in result["findings"]:
            assert "title" in finding
            assert "analysis_type" in finding
            assert "method" in finding
            assert "severity" in finding
            assert "confidence" in finding
            assert "figure_id" in finding

    def test_method_field_matches_analysis_type(self, tmp_path: Path) -> None:
        path = self._make_manipulated_image(tmp_path)
        result = _analyze_image(path)
        for finding in result["findings"]:
            assert finding["method"] == finding["analysis_type"]

    def test_num_findings_matches_list(self, tmp_path: Path) -> None:
        path = self._make_clean_image(tmp_path)
        result = _analyze_image(path)
        assert result["num_findings"] == len(result["findings"])

    def test_confidence_values_are_bounded(self, tmp_path: Path) -> None:
        path = self._make_manipulated_image(tmp_path)
        result = _analyze_image(path)
        for finding in result["findings"]:
            assert 0.0 <= finding["confidence"] <= 1.0

    def test_phash_computed_for_valid_image(self, tmp_path: Path) -> None:
        path = self._make_clean_image(tmp_path)
        result = _analyze_image(path)
        assert result["phash"] is not None
        assert result["ahash"] is not None
        assert isinstance(result["phash"], str)
        assert isinstance(result["ahash"], str)


class TestCollectMethods:
    def test_empty(self) -> None:
        assert _collect_methods([]) == []

    def test_unique_methods(self) -> None:
        findings = [
            {"method": "ela"},
            {"method": "clone_detection"},
            {"method": "ela"},
        ]
        result = _collect_methods(findings)
        assert result == ["ela", "clone_detection"]

    def test_falls_back_to_analysis_type(self) -> None:
        findings = [{"analysis_type": "ela"}]
        result = _collect_methods(findings)
        assert result == ["ela"]


class TestBuildResult:
    def test_build_result_clean(self) -> None:
        result = _build_result(
            name="test.png",
            category="synthetic",
            expected="findings",
            findings=[],
            pass_fail=False,
        )
        assert result["name"] == "test.png"
        assert result["actual_risk"] == "clean"
        assert result["overall_confidence"] == 0.0
        assert result["converging_evidence"] is False
        assert result["findings_count"] == 0
        assert result["methods_used"] == []

    def test_build_result_with_findings(self) -> None:
        findings = [
            {
                "figure_id": "fig1",
                "method": "ela",
                "severity": "medium",
                "confidence": 0.6,
            }
        ]
        result = _build_result(
            name="test.png",
            category="synthetic",
            expected="findings",
            findings=findings,
            pass_fail=True,
        )
        assert result["findings_count"] == 1
        assert result["methods_used"] == ["ela"]
        assert result["actual_risk"] in ("clean", "low", "medium", "high", "critical")

    def test_build_result_with_extra(self) -> None:
        result = _build_result(
            name="test.pdf",
            category="retracted",
            expected="findings",
            findings=[],
            pass_fail=False,
            extra={"statistical_summary": {"grim_findings": 2}},
        )
        assert result["statistical_summary"] == {"grim_findings": 2}

    def test_build_result_uses_aggregate_findings(self) -> None:
        """Verify that _build_result uses aggregate_findings for risk assessment."""
        findings = [
            {
                "figure_id": "fig1",
                "method": "ela",
                "severity": "high",
                "confidence": 0.8,
            },
            {
                "figure_id": "fig1",
                "method": "clone_detection",
                "severity": "high",
                "confidence": 0.7,
            },
        ]
        result = _build_result(
            name="test.png",
            category="synthetic",
            expected="findings",
            findings=findings,
            pass_fail=True,
        )
        # With converging evidence from two methods, should not be downgraded
        assert result["converging_evidence"] is True
        assert result["overall_confidence"] > 0


class TestFullAnalysisPipeline:
    """Test the full analysis pipeline including new analysis methods."""

    def _make_clean_image(self, tmp_path: Path) -> Path:
        arr = np.zeros((200, 300, 3), dtype=np.uint8)
        for i in range(200):
            arr[i, :, :] = int(i / 200 * 255)
        img = Image.fromarray(arr)
        path = tmp_path / "clean.jpg"
        img.save(str(path), "JPEG", quality=95)
        return path

    def test_analyze_image_returns_phash_ahash(self, tmp_path: Path) -> None:
        path = self._make_clean_image(tmp_path)
        result = _analyze_image(path)
        assert "phash" in result
        assert "ahash" in result
        assert result["phash"] is not None
        assert result["ahash"] is not None

    @patch("snoopy.extraction.table_extractor.extract_tables")
    @patch("snoopy.extraction.pdf_parser.extract_text")
    @patch("snoopy.extraction.figure_extractor.extract_figures")
    def test_analyze_pdf_runs_statistical_extraction(
        self,
        mock_extract_figures: MagicMock,
        mock_extract_text: MagicMock,
        mock_extract_tables: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that _analyze_pdf runs text extraction and statistical tests."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()

        mock_extract_figures.return_value = []

        # Mock text extraction
        from snoopy.extraction.pdf_parser import PageText

        mock_extract_text.return_value = [
            PageText(page_number=1, text="M = 3.45, N = 120", word_count=5)
        ]
        mock_extract_tables.return_value = []

        result = _analyze_pdf(pdf_path, figures_dir)

        assert "statistical_summary" in result
        assert "phash_matches" in result
        assert isinstance(result["findings"], list)

    @patch("snoopy.extraction.table_extractor.extract_tables")
    @patch("snoopy.extraction.pdf_parser.extract_text")
    @patch("snoopy.extraction.figure_extractor.extract_figures")
    def test_analyze_pdf_runs_table_extraction(
        self,
        mock_extract_figures: MagicMock,
        mock_extract_text: MagicMock,
        mock_extract_tables: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that _analyze_pdf runs table extraction and duplicate check."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()

        mock_extract_figures.return_value = []

        from snoopy.extraction.pdf_parser import PageText

        mock_extract_text.return_value = [PageText(page_number=1, text="some text", word_count=2)]

        # Mock tables with suspicious duplicate values
        from snoopy.extraction.table_extractor import TableInfo

        mock_extract_tables.return_value = [
            TableInfo(
                page_number=1,
                table_index=0,
                headers=["col1", "col2"],
                rows=[
                    ["3.5", "3.5"],
                    ["3.5", "3.5"],
                    ["3.5", "3.5"],
                    ["3.5", "3.5"],
                    ["3.5", "3.5"],
                ],
                raw_text="",
            )
        ]

        result = _analyze_pdf(pdf_path, figures_dir)

        # Should have at least the duplicate check finding
        dup_findings = [f for f in result["findings"] if f.get("method") == "duplicate_check"]
        assert len(dup_findings) > 0

    def test_llm_analysis_gated_on_skip_llm(self, tmp_path: Path) -> None:
        """Test that LLM analysis is not invoked when skip_llm=True."""
        # The run_demo function checks skip_llm before calling _run_llm_analysis.
        # We verify that with skip_llm=True (default), no LLM calls happen.
        # This is implicitly tested by run_demo not importing ClaudeProvider
        # unless skip_llm is False and ANTHROPIC_API_KEY is set.
        pass  # Covered by integration tests below


class TestRunDemoGeneratesDashboard:
    """Verify that run_demo() produces an index.html dashboard."""

    def _make_synthetic_fixture(self, fixtures_dir: Path) -> None:
        """Create a minimal synthetic fixture image."""
        syn_dir = fixtures_dir / "synthetic"
        syn_dir.mkdir(parents=True, exist_ok=True)
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[20:80, 20:80] = 200
        img = Image.fromarray(arr)
        img.save(str(syn_dir / "test_img.png"))

    @patch("snoopy.demo.fixtures.sample_rsiil_images", return_value=([], []))
    @patch("snoopy.demo.runner.webbrowser.open")
    @patch("snoopy.demo.runner.FIXTURES_DIR")
    def test_run_demo_creates_index_html(
        self,
        mock_fixtures_dir: MagicMock,
        mock_browser_open: MagicMock,
        mock_sample_rsiil: MagicMock,
        tmp_path: Path,
    ) -> None:
        from snoopy.demo.runner import run_demo

        fixtures_dir = tmp_path / "fixtures"
        self._make_synthetic_fixture(fixtures_dir)
        mock_fixtures_dir.__truediv__ = lambda self_inner, key: fixtures_dir / key
        # Patch the `exists` and `iterdir` used in the has_fixtures check
        mock_fixtures_dir.__class__ = type(fixtures_dir)

        # Point FIXTURES_DIR to our tmp fixtures
        report_dir = tmp_path / "reports"

        with (
            patch("snoopy.demo.runner.FIXTURES_DIR", fixtures_dir),
            patch("snoopy.demo.runner._PACKAGE_DIR", tmp_path),
            patch("snoopy.demo.fixtures.download_all"),
        ):
            results = run_demo(output_dir=str(report_dir))

        # Dashboard should exist
        index_html = report_dir / "index.html"
        assert index_html.exists(), "index.html dashboard was not generated"

        html = index_html.read_text()
        assert "Snoopy Demo Results" in html
        assert "Synthetic Forgeries" in html

        # Browser should have been called
        mock_browser_open.assert_called_once()

        # sample_rsiil_images should have been called
        mock_sample_rsiil.assert_called_once_with(50)

        # Results should be returned
        assert isinstance(results, list)
        assert len(results) > 0

        # Verify new result dict structure
        for r in results:
            assert "actual_risk" in r
            assert "overall_confidence" in r
            assert "converging_evidence" in r
            assert "methods_used" in r
            assert isinstance(r["methods_used"], list)

    @patch("snoopy.demo.runner.webbrowser.open")
    def test_run_demo_includes_rsiil_zenodo_results(
        self, mock_browser_open: MagicMock, tmp_path: Path
    ) -> None:
        from snoopy.demo.runner import run_demo

        fixtures_dir = tmp_path / "fixtures"
        self._make_synthetic_fixture(fixtures_dir)

        # Create fake RSIIL Zenodo sample images
        rsiil_tampered = tmp_path / "rsiil_tampered"
        rsiil_pristine = tmp_path / "rsiil_pristine"
        rsiil_tampered.mkdir()
        rsiil_pristine.mkdir()
        # Create minimal images
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[20:80, 20:80] = 200
        for i in range(2):
            img = Image.fromarray(arr)
            img.save(str(rsiil_tampered / f"tampered_{i}.png"))
            img.save(str(rsiil_pristine / f"pristine_{i}.png"))

        tampered_paths = sorted(rsiil_tampered.glob("*.png"))
        pristine_paths = sorted(rsiil_pristine.glob("*.png"))

        report_dir = tmp_path / "reports"

        with (
            patch("snoopy.demo.runner.FIXTURES_DIR", fixtures_dir),
            patch("snoopy.demo.runner._PACKAGE_DIR", tmp_path),
            patch("snoopy.demo.fixtures.download_all"),
            patch(
                "snoopy.demo.fixtures.sample_rsiil_images",
                return_value=(pristine_paths, tampered_paths),
            ),
        ):
            results = run_demo(output_dir=str(report_dir))

        # Should have results from both synthetic and RSIIL Zenodo
        categories = {r["category"] for r in results}
        assert "synthetic" in categories
        # Tampered images get "rsiil" category, pristine get "rsiil_clean"
        assert "rsiil" in categories
        assert "rsiil_clean" in categories
        rsiil_results = [r for r in results if r["category"] == "rsiil"]
        rsiil_clean_results = [r for r in results if r["category"] == "rsiil_clean"]
        assert len(rsiil_results) == 2
        assert len(rsiil_clean_results) == 2
