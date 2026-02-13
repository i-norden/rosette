"""Tests for snoopy.reporting.proof module."""

from __future__ import annotations

import snoopy
from snoopy.reporting.proof import (
    _compute_methods_summary,
    _prepare_findings,
    _prepare_paper_context,
    generate_html_report,
    generate_markdown_report,
)


def _make_paper(**overrides) -> dict:
    defaults = {
        "title": "Test Paper",
        "doi": "10.1234/test",
        "journal": "Test Journal",
        "citation_count": 42,
        "priority_score": 75.0,
        "publication_year": 2024,
        "authors_json": '[{"name": "Alice"}, {"name": "Bob"}]',
    }
    defaults.update(overrides)
    return defaults


def _make_finding(**overrides) -> dict:
    defaults = {
        "figure_id": "fig1",
        "analysis_type": "ela",
        "severity": "medium",
        "confidence": 0.7,
        "title": "Test Finding",
        "description": "Some description",
        "evidence_json": '{"key": "value"}',
        "model_used": "test-model",
    }
    defaults.update(overrides)
    return defaults


class TestPrepareContext:
    def test_basic_paper_context(self):
        ctx = _prepare_paper_context(_make_paper())
        assert ctx["title"] == "Test Paper"
        assert ctx["doi"] == "10.1234/test"
        assert "Alice" in ctx["authors_display"]
        assert "Bob" in ctx["authors_display"]

    def test_missing_authors(self):
        ctx = _prepare_paper_context(_make_paper(authors_json=None))
        assert ctx["authors_display"] == "Unknown"

    def test_many_authors_truncated(self):
        authors = [{"name": f"Author {i}"} for i in range(10)]
        import json

        ctx = _prepare_paper_context(_make_paper(authors_json=json.dumps(authors)))
        assert "et al." in ctx["authors_display"]
        assert "10 total" in ctx["authors_display"]


class TestPrepareFindings:
    def test_findings_sorted_by_severity(self):
        findings = [
            _make_finding(severity="low"),
            _make_finding(severity="critical"),
            _make_finding(severity="medium"),
        ]
        prepared = _prepare_findings(findings, {})
        assert prepared[0]["severity"] == "critical"
        assert prepared[1]["severity"] == "medium"
        assert prepared[2]["severity"] == "low"

    def test_empty_findings(self):
        assert _prepare_findings([], {}) == []


class TestMethodsSummary:
    def test_counts_methods(self):
        findings = [
            _make_finding(analysis_type="ela", severity="medium"),
            _make_finding(analysis_type="ela", severity="low"),
            _make_finding(analysis_type="clone_detection", severity="high"),
        ]
        summary = _compute_methods_summary(findings, total_figures=5)
        by_name = {m["name"]: m for m in summary}
        assert by_name["clone_detection"]["issues_found"] == 1
        assert by_name["ela"]["issues_found"] == 1  # only medium counts


class TestXSSEscaping:
    """Verify that HTML autoescape prevents XSS in generated reports."""

    def test_xss_in_finding_title_escaped(self):
        finding = _make_finding(title='<script>alert("xss")</script>')
        html = generate_html_report(
            paper=_make_paper(),
            findings=[finding],
            figures={},
            summary="Test summary",
            overall_risk="medium",
            overall_confidence=0.5,
            converging_evidence=False,
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_xss_in_description_escaped(self):
        finding = _make_finding(description='<img src=x onerror="alert(1)">')
        html = generate_html_report(
            paper=_make_paper(),
            findings=[finding],
            figures={},
            summary="Safe",
            overall_risk="low",
            overall_confidence=0.3,
            converging_evidence=False,
        )
        assert 'onerror="alert(1)"' not in html

    def test_xss_in_summary_escaped(self):
        html = generate_html_report(
            paper=_make_paper(),
            findings=[],
            figures={},
            summary="<script>steal(cookies)</script>",
            overall_risk="clean",
            overall_confidence=0.0,
            converging_evidence=False,
        )
        assert "<script>" not in html

    def test_markdown_report_does_not_escape(self):
        """Markdown templates should not HTML-escape content."""
        md = generate_markdown_report(
            paper=_make_paper(),
            findings=[_make_finding(title="Normal & Safe")],
            figures={},
            summary="Summary text",
            overall_risk="low",
            overall_confidence=0.3,
            converging_evidence=False,
        )
        # In markdown, & should remain literal, not &amp;
        assert "Normal & Safe" in md


class TestGenerateReports:
    def test_html_report_contains_structure(self):
        html = generate_html_report(
            paper=_make_paper(),
            findings=[_make_finding()],
            figures={"fig1": {"figure_label": "Figure 1"}},
            summary="Executive summary",
            overall_risk="medium",
            overall_confidence=0.7,
            converging_evidence=True,
        )
        assert "<!DOCTYPE html>" in html
        assert "Test Paper" in html
        assert "Executive summary" in html
        assert snoopy.__version__ in html

    def test_markdown_report_contains_structure(self):
        md = generate_markdown_report(
            paper=_make_paper(),
            findings=[_make_finding()],
            figures={"fig1": {"figure_label": "Figure 1"}},
            summary="Summary here",
            overall_risk="high",
            overall_confidence=0.8,
            converging_evidence=False,
        )
        assert "Summary here" in md
        assert snoopy.__version__ in md
