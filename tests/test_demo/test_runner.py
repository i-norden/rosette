"""Tests for the demo runner module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from snoopy.demo.runner import (
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
    def test_expected_findings_pass_when_findings_exist(self):
        assert _determine_pass_fail_expected_findings("high", [{"severity": "high"}]) is True

    def test_expected_findings_fail_when_empty(self):
        assert _determine_pass_fail_expected_findings("clean", []) is False

    def test_expected_clean_pass_when_no_high_findings(self):
        assert _determine_pass_fail_expected_clean([{"severity": "low"}]) is True

    def test_expected_clean_fail_when_high_findings(self):
        assert _determine_pass_fail_expected_clean([{"severity": "high"}]) is False

    def test_expected_clean_fail_when_critical_findings(self):
        assert _determine_pass_fail_expected_clean([{"severity": "critical"}]) is False

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


class TestRunDemo:
    def test_run_demo_download_only(self, tmp_path):
        """Test that download_only exits early."""
        with (
            patch("snoopy.demo.runner.FIXTURES_DIR", tmp_path),
            patch("snoopy.demo.runner.console"),
            patch("snoopy.demo.runner._PACKAGE_DIR", tmp_path),
            patch("snoopy.demo.fixtures.FIXTURES_DIR", tmp_path),
        ):
            # Create expected directories to avoid download
            for d in ["synthetic", "rsiil", "retracted", "survey", "retraction_watch", "clean"]:
                (tmp_path / d).mkdir(exist_ok=True)

            from snoopy.demo.runner import run_demo

            with patch("snoopy.demo.fixtures.download_all"):
                results = run_demo(download_only=True, output_dir=str(tmp_path / "reports"))

        assert results == []
