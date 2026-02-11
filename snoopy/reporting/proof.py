"""Structured proof/evidence report generation using Jinja2 templates."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

import snoopy


TEMPLATE_DIR = Path(__file__).parent / "templates"


def _get_jinja_env() -> Environment:
    return Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=False)


def _prepare_paper_context(paper: dict) -> dict:
    """Prepare paper data for template rendering."""
    authors = []
    if paper.get("authors_json"):
        try:
            authors = json.loads(paper["authors_json"])
        except (json.JSONDecodeError, TypeError):
            pass
    author_names = [a.get("name", "Unknown") for a in authors] if authors else []
    authors_display = ", ".join(author_names[:5])
    if len(author_names) > 5:
        authors_display += f" et al. ({len(author_names)} total)"

    return {
        "title": paper.get("title", "Unknown"),
        "doi": paper.get("doi"),
        "journal": paper.get("journal"),
        "citation_count": paper.get("citation_count", 0),
        "priority_score": paper.get("priority_score", 0),
        "publication_year": paper.get("publication_year"),
        "authors_display": authors_display or "Unknown",
    }


def _prepare_findings(findings: list[dict], figures: dict[str, dict]) -> list[dict]:
    """Prepare findings for template rendering."""
    prepared = []
    for f in findings:
        fig = figures.get(f.get("figure_id", ""), {})
        evidence_details = ""
        if f.get("evidence_json"):
            try:
                evidence = json.loads(f["evidence_json"])
                evidence_details = json.dumps(evidence, indent=2)
            except (json.JSONDecodeError, TypeError):
                evidence_details = str(f.get("evidence_json", ""))

        prepared.append(
            {
                "title": f.get("title", "Untitled finding"),
                "severity": f.get("severity", "info"),
                "analysis_type": f.get("analysis_type", "unknown"),
                "confidence": f.get("confidence", 0.0),
                "description": f.get("description", ""),
                "figure_label": fig.get("figure_label", ""),
                "model_used": f.get("model_used", ""),
                "evidence_details": evidence_details,
            }
        )

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    prepared.sort(key=lambda x: severity_order.get(x["severity"], 5))
    return prepared


def _compute_methods_summary(findings: list[dict], total_figures: int) -> list[dict]:
    """Compute per-method analysis summary."""
    method_counts: Counter[str] = Counter()
    method_issues: Counter[str] = Counter()

    for f in findings:
        atype = f.get("analysis_type", "unknown")
        method_counts[atype] += 1
        if f.get("severity") in ("critical", "high", "medium"):
            method_issues[atype] += 1

    methods = []
    for method_name in sorted(method_counts.keys()):
        methods.append(
            {
                "name": method_name,
                "figures_analyzed": total_figures
                if method_name in ("ela", "clone_detection", "noise_analysis", "llm_vision")
                else method_counts[method_name],
                "issues_found": method_issues.get(method_name, 0),
            }
        )
    return methods


def generate_markdown_report(
    paper: dict,
    findings: list[dict],
    figures: dict[str, dict],
    summary: str,
    overall_risk: str,
    overall_confidence: float,
    converging_evidence: bool,
) -> str:
    """Generate a Markdown integrity report."""
    env = _get_jinja_env()
    template = env.get_template("report.md.j2")

    prepared_findings = _prepare_findings(findings, figures)
    critical_count = sum(1 for f in prepared_findings if f["severity"] == "critical")

    return template.render(
        paper=_prepare_paper_context(paper),
        findings=prepared_findings,
        summary=summary,
        overall_risk=overall_risk,
        overall_confidence=overall_confidence,
        converging_evidence=converging_evidence,
        critical_count=critical_count,
        methods_summary=_compute_methods_summary(findings, len(figures)),
        version=snoopy.__version__,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


def generate_html_report(
    paper: dict,
    findings: list[dict],
    figures: dict[str, dict],
    summary: str,
    overall_risk: str,
    overall_confidence: float,
    converging_evidence: bool,
) -> str:
    """Generate an HTML integrity report."""
    env = _get_jinja_env()
    template = env.get_template("report.html.j2")

    prepared_findings = _prepare_findings(findings, figures)
    critical_count = sum(1 for f in prepared_findings if f["severity"] == "critical")

    return template.render(
        paper=_prepare_paper_context(paper),
        findings=prepared_findings,
        summary=summary,
        overall_risk=overall_risk,
        overall_confidence=overall_confidence,
        converging_evidence=converging_evidence,
        critical_count=critical_count,
        methods_summary=_compute_methods_summary(findings, len(figures)),
        version=snoopy.__version__,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
