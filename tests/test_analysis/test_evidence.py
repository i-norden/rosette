"""Tests for evidence aggregation."""

import pytest

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

    def test_single_method_confidence_capped(self):
        """Single-method findings have confidence capped at max."""
        findings = [
            {
                "figure_id": "fig1",
                "method": "ela",
                "confidence": 0.95,
                "severity": "high",
            }
        ]
        result = aggregate_findings(findings, single_method_max_confidence=0.6)
        fig_ev = result.figure_evidence[0]
        assert fig_ev.confidence <= 0.6

    def test_single_method_custom_max_severity(self):
        """Custom single_method_max_severity is respected."""
        findings = [
            {
                "figure_id": "fig1",
                "method": "ela",
                "confidence": 0.8,
                "severity": "critical",
            }
        ]
        result = aggregate_findings(findings, single_method_max_severity="low")
        fig_ev = result.figure_evidence[0]
        assert fig_ev.severity == "low"


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


class TestMethodWeights:
    """Tests for method_weights parameter in aggregation functions."""

    def test_weighted_confidence(self):
        """Weighted average should differ from simple average."""
        findings = [
            {"confidence": 0.9, "method": "clone_detection"},
            {"confidence": 0.5, "method": "benford"},
        ]
        weights = {"clone_detection": 0.85, "benford": 0.30}

        weighted = compute_overall_confidence(findings, method_weights=weights)
        simple = compute_overall_confidence(findings)

        # Weighted should favor clone_detection more heavily
        assert weighted != simple

    def test_low_weight_method_excluded_from_convergence(self):
        """Methods with weight <= 0.3 should not count for convergence."""
        findings = [
            {"severity": "medium", "method": "clone_detection"},
            {"severity": "medium", "method": "weak_method"},
        ]
        weights = {"clone_detection": 0.85, "weak_method": 0.2}

        severity = compute_figure_severity(findings, method_weights=weights)
        # weak_method doesn't count for convergence so no boost
        assert severity == "medium"

    def test_high_weight_methods_enable_convergence(self):
        """Two high-weight methods should enable convergence boost."""
        findings = [
            {"severity": "medium", "method": "clone_detection"},
            {"severity": "medium", "method": "phash"},
        ]
        weights = {"clone_detection": 0.85, "phash": 0.90}

        severity = compute_figure_severity(findings, method_weights=weights)
        assert severity == "high"  # Boosted from medium

    @pytest.mark.parametrize(
        "weights,expected_risk",
        [
            ({"ela": 0.35, "noise": 0.50}, "low"),  # single figure, single method
            (None, "low"),  # no weights (default behavior)
        ],
    )
    def test_aggregate_with_various_weights(self, weights, expected_risk):
        """Aggregate findings with different weight configs."""
        findings = [
            {"figure_id": "fig1", "method": "ela", "confidence": 0.8, "severity": "medium"},
        ]
        result = aggregate_findings(findings, method_weights=weights)
        assert result.paper_risk == expected_risk

    def test_aggregate_critical_converging_with_weights(self):
        """Critical + converging evidence with weights should set paper_risk to critical."""
        findings = [
            {
                "figure_id": "fig1",
                "method": "clone_detection",
                "confidence": 0.9,
                "severity": "critical",
            },
            {"figure_id": "fig1", "method": "phash", "confidence": 0.95, "severity": "critical"},
        ]
        weights = {"clone_detection": 0.85, "phash": 0.90}
        result = aggregate_findings(findings, method_weights=weights)
        assert result.paper_risk == "critical"
        assert result.converging_evidence is True
        assert result.critical_count >= 1
