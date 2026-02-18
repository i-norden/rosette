"""Evidence aggregation and confidence scoring.

Combines findings from multiple independent analysis methods to produce a
holistic assessment of figure and paper integrity. Converging evidence from
multiple methods strengthens confidence in flagged anomalies, while
single-method flags are appropriately downgraded.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FigureEvidence:
    """Aggregated evidence for a single figure."""

    figure_id: str
    methods_flagged: set[str]
    converging: bool
    severity: str
    confidence: float
    findings: list[dict]
    single_method: bool = False


@dataclass
class AggregatedEvidence:
    """Aggregated evidence across all figures in a paper."""

    paper_risk: str
    overall_confidence: float
    converging_evidence: bool
    figure_evidence: list[FigureEvidence]
    total_findings: int
    critical_count: int


def _get_method(finding: dict) -> str:
    """Extract the method name from a finding dict.

    Handles both 'method' and 'analysis_type' keys for compatibility
    between production orchestrator and demo runner.
    """
    return finding.get("method") or finding.get("analysis_type", "")


def compute_figure_severity(
    findings: list[dict],
    method_weights: dict[str, float] | None = None,
) -> str:
    """Compute the severity level for a set of findings about a single figure.

    Looks at the maximum severity among findings and boosts it if there is
    converging evidence from multiple independent methods.

    Args:
        findings: List of finding dicts, each expected to have at least
            'severity' and 'method' or 'analysis_type' keys.
        method_weights: Optional mapping of method name to weight (0.0-1.0).
            When provided, only methods with weight > 0.3 contribute to
            severity boosting from convergence.

    Returns:
        A severity string: 'critical', 'high', 'medium', 'low', or 'clean'.
    """
    if not findings:
        return "clean"

    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "clean": 0}
    reverse_order = {v: k for k, v in severity_order.items()}

    max_severity = 0
    methods = set()
    for f in findings:
        sev = f.get("severity", "low")
        max_severity = max(max_severity, severity_order.get(sev, 0))
        method = _get_method(f)
        if method:
            # Only count methods with sufficient weight for convergence
            if method_weights is None or method_weights.get(method, 0.5) > 0.3:
                methods.add(method)

    # Boost severity if multiple independent methods agree
    if len(methods) >= 2 and max_severity < 4:
        max_severity = min(max_severity + 1, 4)

    return reverse_order.get(max_severity, "low")


def compute_overall_confidence(
    findings: list[dict],
    method_weights: dict[str, float] | None = None,
) -> float:
    """Compute overall confidence score from individual finding confidences.

    When method_weights are provided, uses a weighted average instead of a
    simple average. The result is boosted by 0.1 if there is converging
    evidence from multiple methods, and capped at 1.0.

    Args:
        findings: List of finding dicts, each expected to have at least
            'confidence' and 'method' or 'analysis_type' keys.
        method_weights: Optional mapping of method name to weight (0.0-1.0).

    Returns:
        Overall confidence score between 0.0 and 1.0.
    """
    if not findings:
        return 0.0

    methods = set()
    for f in findings:
        m = _get_method(f)
        if m:
            methods.add(m)

    if method_weights:
        total_weight = 0.0
        weighted_sum = 0.0
        for f in findings:
            conf = float(f.get("confidence", 0.0))
            method = _get_method(f)
            weight = method_weights.get(method, 0.5)
            weighted_sum += conf * weight
            total_weight += weight
        weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0.0
    else:
        confidences = [float(f.get("confidence", 0.0)) for f in findings]
        weighted_avg = sum(confidences) / len(confidences)

    # Boost if converging evidence
    if len(methods) >= 2:
        weighted_avg += 0.1

    return min(weighted_avg, 1.0)


def aggregate_findings(
    findings: list[dict],
    method_weights: dict[str, float] | None = None,
    single_method_max_severity: str = "medium",
    single_method_max_confidence: float = 0.7,
) -> AggregatedEvidence:
    """Aggregate findings across all analysis methods into a unified assessment.

    Groups findings by figure, determines convergence, computes severity and
    confidence, and assigns an overall paper risk level.

    Each finding dict is expected to contain at least:
        - figure_id (str): Identifier for the figure.
        - method or analysis_type (str): Analysis method that produced the finding.
        - confidence (float): Confidence in this finding (0.0 to 1.0).
        - severity (str): Severity level of the finding.

    Args:
        findings: List of finding dictionaries from all analysis methods.
        method_weights: Optional mapping of method name to weight (0.0-1.0).
            Used for weighted confidence averaging.

    Returns:
        AggregatedEvidence with per-figure and overall assessments.
    """
    if not findings:
        return AggregatedEvidence(
            paper_risk="clean",
            overall_confidence=0.0,
            converging_evidence=False,
            figure_evidence=[],
            total_findings=0,
            critical_count=0,
        )

    # Group findings by figure_id.
    # Findings without a figure_id get unique keys so unrelated paper-level
    # findings (GRIM, Benford, etc.) are not incorrectly grouped together.
    figure_groups: dict[str, list[dict]] = {}
    _ungrouped_counter = 0
    for f in findings:
        fig_id = f.get("figure_id") or ""
        if not fig_id:
            fig_id = f"_paper_finding_{_ungrouped_counter}"
            _ungrouped_counter += 1
        figure_groups.setdefault(fig_id, []).append(f)

    figure_evidence_list: list[FigureEvidence] = []
    any_converging = False
    critical_count = 0
    figures_flagged = 0
    has_critical_converging = False

    for fig_id, fig_findings in figure_groups.items():
        # Determine which independent methods flagged this figure
        # Only count methods where confidence exceeds 0.6
        methods_flagged: set[str] = set()
        for f in fig_findings:
            confidence = float(f.get("confidence", 0.0))
            method = _get_method(f)
            if confidence > 0.6 and method:
                methods_flagged.add(method)

        converging = len(methods_flagged) >= 2
        if converging:
            any_converging = True

        severity = compute_figure_severity(fig_findings, method_weights=method_weights)
        confidence = compute_overall_confidence(fig_findings, method_weights=method_weights)

        # For single-method flags, preserve original severity but mark them
        # and apply configurable confidence cap (default 0.7, softer than old 0.5)
        is_single_method = not converging and len(methods_flagged) > 0
        if is_single_method:
            severity_order_local = {"critical": 4, "high": 3, "medium": 2, "low": 1, "clean": 0}
            max_sev_val = severity_order_local.get(single_method_max_severity, 2)
            cur_sev_val = severity_order_local.get(severity, 0)
            if cur_sev_val > max_sev_val:
                reverse_local = {v: k for k, v in severity_order_local.items()}
                severity = reverse_local.get(max_sev_val, single_method_max_severity)
            confidence = min(confidence, single_method_max_confidence)

        if severity == "critical":
            critical_count += 1
            if converging:
                has_critical_converging = True

        if methods_flagged:
            figures_flagged += 1

        figure_evidence_list.append(
            FigureEvidence(
                figure_id=fig_id,
                methods_flagged=methods_flagged,
                converging=converging,
                severity=severity,
                confidence=confidence,
                findings=fig_findings,
                single_method=is_single_method,
            )
        )

    # Determine overall paper risk
    if has_critical_converging:
        paper_risk = "critical"
    elif any_converging and figures_flagged >= 2:
        paper_risk = "high"
    elif figures_flagged >= 3 and not any_converging:
        paper_risk = "medium"
    elif figures_flagged == 1 and any_converging:
        paper_risk = "medium"
    elif figures_flagged >= 1:
        paper_risk = "low"
    else:
        paper_risk = "clean"

    overall_confidence = compute_overall_confidence(findings, method_weights=method_weights)

    return AggregatedEvidence(
        paper_risk=paper_risk,
        overall_confidence=overall_confidence,
        converging_evidence=any_converging,
        figure_evidence=figure_evidence_list,
        total_findings=len(findings),
        critical_count=critical_count,
    )
