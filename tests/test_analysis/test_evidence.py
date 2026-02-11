"""Tests for evidence aggregation."""

from snoopy.analysis.evidence import (
    aggregate_findings,
    compute_figure_severity,
    compute_overall_confidence,
)


class TestAggregateFindingsEmpty:
    def test_empty_findings(self):
        result = aggregate_findings([])
        assert result.paper_risk == "clean"
        assert result.total_findings == 0
        assert result.converging_evidence is False


class TestSingleMethod:
    def test_single_method_downgraded(self):
        findings = [
            {
                "figure_id": "fig1",
                "method": "ela",
                "confidence": 0.8,
                "severity": "high",
            }
        ]
        result = aggregate_findings(findings)
        # Single method should be downgraded
        assert result.paper_risk == "low"
        assert result.converging_evidence is False


class TestConvergingEvidence:
    def test_two_methods_converge(self):
        findings = [
            {
                "figure_id": "fig1",
                "method": "ela",
                "confidence": 0.8,
                "severity": "high",
            },
            {
                "figure_id": "fig1",
                "method": "clone_detection",
                "confidence": 0.7,
                "severity": "high",
            },
        ]
        result = aggregate_findings(findings)
        assert result.converging_evidence is True
        assert result.paper_risk in ("medium", "high", "critical")

    def test_multiple_figures_flagged(self):
        findings = [
            {"figure_id": "fig1", "method": "ela", "confidence": 0.8, "severity": "medium"},
            {"figure_id": "fig1", "method": "llm", "confidence": 0.7, "severity": "medium"},
            {"figure_id": "fig2", "method": "ela", "confidence": 0.8, "severity": "medium"},
            {"figure_id": "fig2", "method": "noise", "confidence": 0.7, "severity": "medium"},
        ]
        result = aggregate_findings(findings)
        assert result.paper_risk == "high"


class TestSeverity:
    def test_clean_with_no_findings(self):
        assert compute_figure_severity([]) == "clean"

    def test_single_finding(self):
        result = compute_figure_severity([{"severity": "medium", "method": "ela"}])
        assert result == "medium"

    def test_boosted_with_multiple_methods(self):
        result = compute_figure_severity(
            [
                {"severity": "medium", "method": "ela"},
                {"severity": "medium", "method": "clone"},
            ]
        )
        assert result == "high"  # Boosted from medium


class TestConfidence:
    def test_empty(self):
        assert compute_overall_confidence([]) == 0.0

    def test_single_finding(self):
        result = compute_overall_confidence([{"confidence": 0.8, "method": "ela"}])
        assert abs(result - 0.8) < 0.01

    def test_boost_with_convergence(self):
        result = compute_overall_confidence(
            [
                {"confidence": 0.7, "method": "ela"},
                {"confidence": 0.7, "method": "clone"},
            ]
        )
        # Average 0.7 + 0.1 boost = 0.8
        assert abs(result - 0.8) < 0.01
