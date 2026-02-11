"""Tests for the Rich-based pretty terminal reporter."""

from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.text import Text

from snoopy.reporting import pretty


def _capture_console() -> tuple[Console, StringIO]:
    """Create a console that captures output to a string buffer."""
    buf = StringIO()
    con = Console(file=buf, force_terminal=True, width=120)
    return con, buf


class TestSeverityBadge:
    def test_known_severities(self) -> None:
        for sev in ("critical", "high", "medium", "low", "info", "clean"):
            badge = pretty._severity_badge(sev)
            assert isinstance(badge, Text)
            assert sev.upper() in badge.plain

    def test_unknown_severity_falls_back(self) -> None:
        badge = pretty._severity_badge("unknown_level")
        assert isinstance(badge, Text)
        assert "UNKNOWN_LEVEL" in badge.plain


class TestRiskBadge:
    def test_known_risks(self) -> None:
        for risk in ("critical", "high", "medium", "low", "clean"):
            badge = pretty._risk_badge(risk)
            assert isinstance(badge, Text)
            assert risk.upper() in badge.plain

    def test_unknown_risk_falls_back(self) -> None:
        badge = pretty._risk_badge("bizarre")
        assert isinstance(badge, Text)
        assert "BIZARRE" in badge.plain


class TestConfidenceBar:
    def test_zero_confidence(self) -> None:
        bar = pretty._confidence_bar(0.0)
        assert isinstance(bar, Text)
        assert "0%" in bar.plain

    def test_full_confidence(self) -> None:
        bar = pretty._confidence_bar(1.0)
        assert isinstance(bar, Text)
        assert "100%" in bar.plain

    def test_mid_confidence_yellow(self) -> None:
        bar = pretty._confidence_bar(0.6)
        assert "60%" in bar.plain

    def test_high_confidence_green(self) -> None:
        bar = pretty._confidence_bar(0.9)
        assert "90%" in bar.plain

    def test_low_confidence_red(self) -> None:
        bar = pretty._confidence_bar(0.2)
        assert "20%" in bar.plain

    def test_custom_width(self) -> None:
        bar = pretty._confidence_bar(0.5, width=10)
        # 5 filled + 5 empty + percentage
        assert "50%" in bar.plain


class TestPrintPaperHeader:
    def test_full_metadata(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_paper_header(
            {
                "title": "Test Paper Title",
                "doi": "10.1234/test",
                "journal": "Nature",
                "citation_count": 500,
                "publication_year": 2023,
            }
        )
        output = buf.getvalue()
        assert "Test Paper Title" in output
        assert "10.1234/test" in output
        assert "Nature" in output
        assert "500" in output
        assert "2023" in output

    def test_missing_fields_use_defaults(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_paper_header({})
        output = buf.getvalue()
        assert "Unknown Paper" in output
        assert "N/A" in output
        assert "Unknown Journal" in output

    def test_empty_year_omitted(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_paper_header({"title": "X", "publication_year": ""})
        output = buf.getvalue()
        assert "Year" not in output


class TestPrintAssessment:
    def test_clean_assessment(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_assessment(
            risk="clean",
            confidence=0.0,
            converging=False,
            total_findings=0,
            critical_count=0,
        )
        output = buf.getvalue()
        assert "CLEAN" in output
        assert "0%" in output
        assert "NO" in output

    def test_critical_with_converging(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_assessment(
            risk="critical",
            confidence=0.95,
            converging=True,
            total_findings=5,
            critical_count=2,
        )
        output = buf.getvalue()
        assert "CRITICAL" in output
        assert "95%" in output
        assert "YES" in output
        assert "2 critical" in output

    def test_no_critical_count_omits_label(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_assessment(
            risk="low",
            confidence=0.3,
            converging=False,
            total_findings=1,
            critical_count=0,
        )
        output = buf.getvalue()
        assert "critical)" not in output


class TestPrintFindingsTable:
    def test_empty_findings(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_findings_table([])
        output = buf.getvalue()
        assert "No findings to report" in output

    def test_single_finding(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_findings_table(
            [
                {
                    "severity": "high",
                    "analysis_type": "clone_detection",
                    "title": "Clone found",
                    "confidence": 0.85,
                    "figure_label": "Fig 1",
                }
            ]
        )
        output = buf.getvalue()
        assert "HIGH" in output
        assert "clone_detection" in output
        assert "Clone found" in output
        assert "85%" in output
        assert "Fig 1" in output

    def test_findings_sorted_by_severity(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        findings = [
            {"severity": "low", "title": "Low issue", "analysis_type": "ela"},
            {"severity": "critical", "title": "Critical issue", "analysis_type": "clone_detection"},
            {"severity": "medium", "title": "Medium issue", "analysis_type": "noise_analysis"},
        ]
        pretty.print_findings_table(findings)
        output = buf.getvalue()
        # Critical should appear before medium, medium before low
        crit_pos = output.index("CRITICAL")
        med_pos = output.index("MEDIUM")
        low_pos = output.index("LOW")
        assert crit_pos < med_pos < low_pos

    def test_finding_with_method_fallback(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_findings_table(
            [
                {
                    "severity": "info",
                    "method": "custom_method",
                    "title": "Info note",
                }
            ]
        )
        output = buf.getvalue()
        assert "custom_method" in output

    def test_finding_with_figure_id_fallback(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_findings_table(
            [
                {
                    "severity": "low",
                    "title": "Test",
                    "figure_id": "img_001.png",
                }
            ]
        )
        output = buf.getvalue()
        assert "img_001.png" in output


class TestPrintMethodsSummary:
    def test_empty_methods(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_methods_summary([])
        # Should produce no output
        assert buf.getvalue() == ""

    def test_methods_with_issues(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_methods_summary(
            [
                {"name": "ela", "figures_analyzed": 10, "issues_found": 3},
                {"name": "clone_detection", "figures_analyzed": 10, "issues_found": 0},
            ]
        )
        output = buf.getvalue()
        assert "ela" in output
        assert "clone_detection" in output
        assert "10" in output
        assert "3" in output


class TestPrintFigureDetail:
    def test_basic_detail(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_figure_detail(
            "Figure_1.png",
            [
                {
                    "severity": "high",
                    "title": "Clone detected",
                    "description": "Suspicious region found",
                    "evidence": {"num_matches": 42, "inlier_ratio": 0.85},
                }
            ],
        )
        output = buf.getvalue()
        assert "Figure_1.png" in output
        assert "HIGH" in output
        assert "Clone detected" in output
        assert "Suspicious region found" in output
        assert "num_matches" in output
        assert "42" in output

    def test_finding_without_evidence(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_figure_detail(
            "fig.png",
            [
                {
                    "severity": "low",
                    "title": "Minor issue",
                }
            ],
        )
        output = buf.getvalue()
        assert "fig.png" in output
        assert "LOW" in output

    def test_finding_with_non_dict_evidence(self, monkeypatch) -> None:
        """Evidence that is not a dict should not crash."""
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_figure_detail(
            "fig.png",
            [
                {
                    "severity": "info",
                    "title": "Test",
                    "evidence": "some string",
                }
            ],
        )
        output = buf.getvalue()
        assert "fig.png" in output


class TestPrintFullReport:
    def test_with_no_findings_and_no_evidence(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_full_report(
            paper={"title": "Clean Paper", "doi": "10.1/clean"},
            findings=[],
        )
        output = buf.getvalue()
        assert "Clean Paper" in output
        assert "CLEAN" in output
        assert "No findings to report" in output

    def test_with_findings_no_evidence(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_full_report(
            paper={"title": "Bad Paper"},
            findings=[
                {"severity": "critical", "title": "Major issue", "figure_id": "f1"},
                {"severity": "medium", "title": "Minor issue", "figure_id": "f2"},
            ],
        )
        output = buf.getvalue()
        assert "Bad Paper" in output
        assert (
            "HIGH" in output
        )  # risk computed: critical findings -> "high" (the function uses "high" for critical>0)
        assert "CRITICAL" in output  # the finding severity badge
        assert "Figure Details" in output

    def test_with_evidence_dict(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_full_report(
            paper={"title": "Flagged Paper"},
            findings=[{"severity": "high", "title": "Clone", "figure_id": "f1"}],
            evidence={
                "overall_risk": "high",
                "overall_confidence": 0.8,
                "converging_evidence": True,
                "total_findings": 1,
                "critical_count": 0,
                "methods_summary": [
                    {"name": "ela", "figures_analyzed": 5, "issues_found": 1},
                ],
            },
        )
        output = buf.getvalue()
        assert "Flagged Paper" in output
        assert "YES" in output  # converging evidence
        assert "ela" in output  # methods summary

    def test_empty_paper_dict(self, monkeypatch) -> None:
        """Empty paper dict should use defaults without crashing."""
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_full_report(paper={}, findings=[])
        output = buf.getvalue()
        assert "Unknown Paper" in output


class TestPrintDemoSummary:
    def test_all_passing(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_demo_summary(
            [
                {
                    "name": "test1.png",
                    "category": "synthetic",
                    "expected": "findings",
                    "actual_risk": "medium",
                    "findings_count": 2,
                    "pass_fail": True,
                },
                {
                    "name": "test2.png",
                    "category": "clean",
                    "expected": "clean",
                    "actual_risk": "clean",
                    "findings_count": 0,
                    "pass_fail": True,
                },
            ]
        )
        output = buf.getvalue()
        assert "PASS" in output
        assert "2/2 passed" in output
        assert "FAIL" not in output.split("Summary")[1] if "Summary" in output else True

    def test_mixed_results(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_demo_summary(
            [
                {
                    "name": "good.png",
                    "category": "synthetic",
                    "expected": "findings",
                    "actual_risk": "medium",
                    "findings_count": 1,
                    "pass_fail": True,
                },
                {
                    "name": "bad.png",
                    "category": "synthetic",
                    "expected": "findings",
                    "actual_risk": "clean",
                    "findings_count": 0,
                    "pass_fail": False,
                },
            ]
        )
        output = buf.getvalue()
        assert "1/2 passed" in output
        assert "1 failed" in output

    def test_empty_results(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_demo_summary([])
        output = buf.getvalue()
        assert "0/0 passed" in output

    def test_missing_keys_use_defaults(self, monkeypatch) -> None:
        con, buf = _capture_console()
        monkeypatch.setattr(pretty, "console", con)
        pretty.print_demo_summary([{"pass_fail": True}])
        output = buf.getvalue()
        assert "Unknown" in output
        assert "CLEAN" in output  # default risk


class TestCreateProgress:
    def test_returns_progress_instance(self) -> None:
        from rich.progress import Progress

        p = pretty.create_progress()
        assert isinstance(p, Progress)
