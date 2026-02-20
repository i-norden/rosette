"""Tests for the demo runner module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from rosette.demo.runner import (
    _build_result,
    _collect_methods,
    _determine_pass_fail_expected_clean,
    _determine_pass_fail_expected_findings,
    _find_images,
    _find_pdfs,
)


class TestFindImages:
    def test_finds_image_files(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"\x89PNG")
        (tmp_path / "b.jpg").write_bytes(b"\xff\xd8")
        (tmp_path / "c.txt").write_text("not an image")
        images = _find_images(tmp_path)
        assert len(images) == 2
        names = {p.name for p in images}
        assert "a.png" in names
        assert "b.jpg" in names

    def test_returns_empty_for_missing_dir(self):
        assert _find_images(Path("/nonexistent")) == []

    def test_finds_various_extensions(self, tmp_path):
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
            (tmp_path / f"img{ext}").write_bytes(b"\x00")
        images = _find_images(tmp_path)
        assert len(images) == 6


class TestFindPdfs:
    def test_finds_pdf_files(self, tmp_path):
        (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
        (tmp_path / "b.pdf").write_bytes(b"%PDF-1.5")
        (tmp_path / "c.txt").write_text("not a pdf")
        pdfs = _find_pdfs(tmp_path)
        assert len(pdfs) == 2

    def test_returns_empty_for_missing_dir(self):
        assert _find_pdfs(Path("/nonexistent")) == []


class TestCollectMethods:
    def test_collects_unique_methods(self):
        findings = [
            {"method": "ela"},
            {"method": "clone_detection"},
            {"method": "ela"},
        ]
        methods = _collect_methods(findings)
        assert methods == ["ela", "clone_detection"]

    def test_falls_back_to_analysis_type(self):
        findings = [{"analysis_type": "grim"}]
        methods = _collect_methods(findings)
        assert methods == ["grim"]

    def test_empty_findings(self):
        assert _collect_methods([]) == []


class TestPassFailDetermination:
    def test_expected_findings_pass_when_risk_medium_or_above(self):
        assert _determine_pass_fail_expected_findings("medium", [{"severity": "medium"}]) is True
        assert _determine_pass_fail_expected_findings("high", [{"severity": "high"}]) is True
        assert (
            _determine_pass_fail_expected_findings("critical", [{"severity": "critical"}]) is True
        )

    def test_expected_findings_fail_when_risk_low_or_clean(self):
        # Low/clean risk should NOT count as "detected" even if findings exist
        assert _determine_pass_fail_expected_findings("low", [{"severity": "low"}]) is False
        assert _determine_pass_fail_expected_findings("clean", [{"severity": "low"}]) is False

    def test_expected_findings_fail_when_empty(self):
        assert _determine_pass_fail_expected_findings("clean", []) is False

    def test_expected_clean_pass_when_no_high_findings(self):
        assert _determine_pass_fail_expected_clean([{"severity": "low"}]) is True

    def test_expected_clean_pass_single_method_high_finding(self):
        # A single raw "high" finding from one method now passes because
        # aggregation downgrades single-method findings to max "medium"
        findings = [
            {
                "severity": "high",
                "method": "noise_analysis",
                "confidence": 0.8,
                "figure_id": "fig1",
            }
        ]
        assert _determine_pass_fail_expected_clean(findings) is True

    def test_expected_clean_fail_when_converging_high_findings(self):
        # Converging high-severity findings from multiple methods still fail
        findings = [
            {
                "severity": "high",
                "method": "ela",
                "confidence": 0.8,
                "figure_id": "fig1",
            },
            {
                "severity": "high",
                "method": "clone_detection",
                "confidence": 0.8,
                "figure_id": "fig1",
            },
        ]
        assert _determine_pass_fail_expected_clean(findings) is False

    def test_expected_clean_pass_when_no_findings(self):
        assert _determine_pass_fail_expected_clean([]) is True


class TestBuildResult:
    def test_builds_basic_result(self):
        findings = [
            {
                "method": "ela",
                "analysis_type": "ela",
                "severity": "medium",
                "confidence": 0.7,
            }
        ]
        result = _build_result(
            name="test.png",
            category="synthetic",
            expected="findings",
            findings=findings,
            pass_fail=True,
        )
        assert result["name"] == "test.png"
        assert result["category"] == "synthetic"
        assert result["expected"] == "findings"
        assert result["pass_fail"] is True
        assert result["findings_count"] >= 0
        assert isinstance(result["methods_used"], list)

    def test_builds_result_with_extra(self):
        result = _build_result(
            name="test.pdf",
            category="retracted",
            expected="findings",
            findings=[],
            pass_fail=False,
            extra={"statistical_summary": {"grim_findings": 2}},
        )
        assert result["statistical_summary"]["grim_findings"] == 2


class TestAnalyzePdfTextAnalyses:
    """Tests that _analyze_pdf produces text-based analysis findings."""

    def test_analyze_pdf_produces_statistical_findings(self, tmp_path):
        """_analyze_pdf produces GRIMMER, SPRITE, and tortured phrase findings."""
        from rosette.demo.runner import _analyze_pdf

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()

        # Mock figure extraction to return no figures (focus on text analyses)
        # Mock text extraction to return text
        mock_page = type("Page", (), {"text": "Sample text M=3.5, SD=1.2, N=10"})()

        grimmer_finding = {
            "title": "GRIMMER inconsistency",
            "analysis_type": "grimmer",
            "method": "grimmer",
            "severity": "high",
            "confidence": 0.8,
            "description": "SD inconsistent",
            "figure_id": "",
            "evidence": {},
        }
        sprite_finding = {
            "title": "SPRITE inconsistency",
            "analysis_type": "sprite",
            "method": "sprite",
            "severity": "high",
            "confidence": 0.8,
            "description": "Mean not achievable",
            "figure_id": "",
            "evidence": {},
        }
        tp_finding = {
            "title": "Tortured phrases detected (3 unique)",
            "analysis_type": "tortured_phrases",
            "method": "tortured_phrases",
            "severity": "high",
            "confidence": 0.85,
            "description": "Found tortured phrases",
            "figure_id": "",
            "evidence": {},
        }

        from rosette.extraction.stats_extractor import MeanReport

        mean_report = MeanReport(mean=3.5, sd=1.2, n=10, context="test")

        with (
            patch(
                "rosette.extraction.figure_extractor.extract_figures",
                return_value=[],
            ),
            patch(
                "rosette.extraction.pdf_parser.extract_text",
                return_value=[mock_page],
            ),
            patch(
                "rosette.analysis.statistical.grim_test",
                return_value=type("R", (), {"consistent": True})(),
            ),
            patch(
                "rosette.extraction.stats_extractor.extract_means_and_ns",
                return_value=[],
            ),
            patch(
                "rosette.extraction.stats_extractor.extract_test_statistics",
                return_value=[],
            ),
            patch(
                "rosette.extraction.stats_extractor.extract_numerical_values",
                return_value=[],
            ),
            patch(
                "rosette.extraction.stats_extractor.extract_p_values",
                return_value=[],
            ),
            patch(
                "rosette.extraction.table_extractor.extract_tables",
                return_value=[],
            ),
            patch(
                "rosette.analysis.run_analysis.run_statistical_tests",
                return_value=[grimmer_finding],
            ),
            patch(
                "rosette.extraction.stats_extractor.extract_means_sds_and_ns",
                return_value=[mean_report],
            ),
            patch(
                "rosette.analysis.run_analysis.run_sprite_analysis",
                return_value=[sprite_finding],
            ),
            patch(
                "rosette.analysis.run_analysis.run_tortured_phrases",
                return_value=[tp_finding],
            ),
        ):
            result = _analyze_pdf(pdf_path, figures_dir)

        # Verify findings from all three text-based analyses are present
        analysis_types = {f["analysis_type"] for f in result["findings"]}
        assert "grimmer" in analysis_types
        assert "sprite" in analysis_types
        assert "tortured_phrases" in analysis_types
        assert result["num_findings"] >= 3

        # Verify statistical summary includes new counts
        summary = result.get("statistical_summary", {})
        assert summary.get("grimmer_findings", 0) >= 1
        assert summary.get("sprite_findings", 0) >= 1
        assert summary.get("tortured_phrase_findings", 0) >= 1


class TestRunDemo:
    def test_run_demo_download_only(self, tmp_path):
        """Test that download_only exits early."""
        # Create fake RSIIL data dirs so auto-download doesn't trigger
        rsiil_data = tmp_path / "data" / "rsiil"
        (rsiil_data / "pristine").mkdir(parents=True)
        (rsiil_data / "test").mkdir(parents=True)
        (rsiil_data / "pristine" / "dummy.png").write_bytes(b"\x89PNG")
        (rsiil_data / "test" / "dummy.png").write_bytes(b"\x89PNG")

        with (
            patch("rosette.demo.runner.FIXTURES_DIR", tmp_path),
            patch("rosette.demo.runner.console"),
            patch("rosette.demo.runner._PACKAGE_DIR", tmp_path),
            patch("rosette.demo.fixtures.FIXTURES_DIR", tmp_path),
            patch("rosette.demo.fixtures.RSIIL_DATA_DIR", rsiil_data),
        ):
            # Create expected directories to avoid download
            for d in ["synthetic", "rsiil", "retracted", "survey", "retraction_watch", "clean"]:
                (tmp_path / d).mkdir(exist_ok=True)

            from rosette.demo.runner import run_demo

            with patch("rosette.demo.fixtures.download_all"):
                results = run_demo(download_only=True, output_dir=str(tmp_path / "reports"))

        assert results == []
