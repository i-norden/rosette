"""Tests for snoopy.reporting.dashboard."""

from __future__ import annotations

from pathlib import Path

from snoopy.reporting.dashboard import generate_dashboard


def _make_results() -> list[dict]:
    """Build a representative list of demo results."""
    return [
        {
            "name": "img_spliced.png",
            "category": "synthetic",
            "expected": "findings",
            "actual_risk": "high",
            "findings_count": 2,
            "pass_fail": True,
            "findings": [
                {
                    "title": "ELA anomaly",
                    "analysis_type": "ela",
                    "severity": "medium",
                    "confidence": 0.7,
                },
                {
                    "title": "Clone detected",
                    "analysis_type": "clone_detection",
                    "severity": "high",
                    "confidence": 0.9,
                },
            ],
        },
        {
            "name": "img_clean.png",
            "category": "synthetic",
            "expected": "findings",
            "actual_risk": "clean",
            "findings_count": 0,
            "pass_fail": False,
            "findings": [],
        },
        {
            "name": "control.pdf",
            "category": "clean",
            "expected": "clean",
            "actual_risk": "clean",
            "findings_count": 0,
            "pass_fail": True,
            "findings": [],
        },
        {
            "name": "survey.pdf",
            "category": "survey",
            "expected": "informational",
            "actual_risk": "medium",
            "findings_count": 1,
            "pass_fail": True,
            "findings": [
                {
                    "title": "Noise anomaly",
                    "analysis_type": "noise_analysis",
                    "severity": "medium",
                    "confidence": 0.5,
                },
            ],
        },
    ]


class TestGenerateDashboard:
    def test_creates_index_html(self, tmp_path: Path) -> None:
        results = _make_results()
        out = generate_dashboard(results, tmp_path)
        assert out == tmp_path / "index.html"
        assert out.exists()

    def test_html_contains_title(self, tmp_path: Path) -> None:
        results = _make_results()
        generate_dashboard(results, tmp_path)
        html = (tmp_path / "index.html").read_text()
        assert "Snoopy Demo Results" in html

    def test_html_contains_category_names(self, tmp_path: Path) -> None:
        results = _make_results()
        generate_dashboard(results, tmp_path)
        html = (tmp_path / "index.html").read_text()
        assert "Synthetic Forgeries" in html
        assert "Clean Controls" in html
        assert "Survey / Informational" in html

    def test_html_contains_result_names(self, tmp_path: Path) -> None:
        results = _make_results()
        generate_dashboard(results, tmp_path)
        html = (tmp_path / "index.html").read_text()
        assert "img_spliced.png" in html
        assert "img_clean.png" in html
        assert "control.pdf" in html

    def test_report_links_are_relative(self, tmp_path: Path) -> None:
        results = _make_results()
        generate_dashboard(results, tmp_path)
        html = (tmp_path / "index.html").read_text()
        assert "./img_spliced_report.html" in html

    def test_no_report_link_for_zero_findings(self, tmp_path: Path) -> None:
        results = _make_results()
        generate_dashboard(results, tmp_path)
        html = (tmp_path / "index.html").read_text()
        # img_clean.png has 0 findings -> no report link
        assert "./img_clean_report.html" not in html

    def test_empty_results(self, tmp_path: Path) -> None:
        out = generate_dashboard([], tmp_path)
        assert out.exists()
        html = out.read_text()
        assert "Snoopy Demo Results" in html

    def test_creates_report_dir_if_missing(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "sub" / "dir"
        out = generate_dashboard(_make_results(), out_dir)
        assert out.exists()

    def test_pass_rate_displayed(self, tmp_path: Path) -> None:
        results = _make_results()
        generate_dashboard(results, tmp_path)
        html = (tmp_path / "index.html").read_text()
        # 3 out of 4 pass -> 75%
        assert "75%" in html

    def test_methods_in_table(self, tmp_path: Path) -> None:
        results = _make_results()
        generate_dashboard(results, tmp_path)
        html = (tmp_path / "index.html").read_text()
        assert "ela" in html
        assert "clone_detection" in html

    def test_findings_expected_shows_detected_label(self, tmp_path: Path) -> None:
        results = _make_results()
        generate_dashboard(results, tmp_path)
        html = (tmp_path / "index.html").read_text()
        assert "Detected?" in html
        assert "DETECTED" in html
        assert "MISSED" in html

    def test_clean_shows_false_positive_label(self, tmp_path: Path) -> None:
        results = _make_results()
        generate_dashboard(results, tmp_path)
        html = (tmp_path / "index.html").read_text()
        assert "False Positive?" in html
        assert "CLEAN" in html

    def test_category_explanations_present(self, tmp_path: Path) -> None:
        results = _make_results()
        generate_dashboard(results, tmp_path)
        html = (tmp_path / "index.html").read_text()
        assert "Goal: detect manipulation" in html
        assert "Goal: avoid false alarms" in html
