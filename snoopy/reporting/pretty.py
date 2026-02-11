"""Rich-based pretty terminal reporter for snoopy analysis results."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

console = Console()

SEVERITY_STYLES = {
    "critical": "bold red",
    "high": "bold bright_red",
    "medium": "bold yellow",
    "low": "bold green",
    "info": "dim",
    "clean": "bold green",
}

RISK_STYLES = {
    "critical": "bold white on red",
    "high": "bold white on bright_red",
    "medium": "bold black on yellow",
    "low": "bold white on green",
    "clean": "bold white on green",
}


def _severity_badge(severity: str) -> Text:
    """Return a styled severity badge."""
    style = SEVERITY_STYLES.get(severity, "dim")
    return Text(f" {severity.upper()} ", style=style)


def _risk_badge(risk: str) -> Text:
    """Return a styled risk level badge."""
    style = RISK_STYLES.get(risk, "dim")
    return Text(f" {risk.upper()} ", style=style)


def _confidence_bar(confidence: float, width: int = 20) -> Text:
    """Return a text-based confidence bar."""
    filled = int(confidence * width)
    empty = width - filled
    if confidence >= 0.8:
        color = "green"
    elif confidence >= 0.5:
        color = "yellow"
    else:
        color = "red"
    bar = Text()
    bar.append("█" * filled, style=color)
    bar.append("░" * empty, style="dim")
    bar.append(f" {confidence:.0%}", style="bold")
    return bar


def print_paper_header(paper: dict) -> None:
    """Print a bordered header panel with paper metadata."""
    title = paper.get("title", "Unknown Paper")
    doi = paper.get("doi", "N/A")
    journal = paper.get("journal", "Unknown Journal")
    citations = paper.get("citation_count", 0)
    year = paper.get("publication_year", "")

    content = Text()
    content.append(f"{title}\n\n", style="bold white")
    content.append("DOI: ", style="dim")
    content.append(f"{doi}\n", style="cyan")
    content.append("Journal: ", style="dim")
    content.append(f"{journal}\n", style="white")
    if year:
        content.append("Year: ", style="dim")
        content.append(f"{year}\n", style="white")
    content.append("Citations: ", style="dim")
    content.append(str(citations), style="bold cyan")

    console.print(Panel(content, title="[bold]Paper Under Analysis[/bold]", border_style="blue"))


def print_assessment(
    risk: str,
    confidence: float,
    converging: bool,
    total_findings: int,
    critical_count: int,
) -> None:
    """Print overall assessment table with risk level and confidence."""
    table = Table(title="Overall Assessment", border_style="bright_blue", show_header=False)
    table.add_column("Metric", style="dim", width=24)
    table.add_column("Value")

    table.add_row("Risk Level", _risk_badge(risk))
    table.add_row("Confidence", _confidence_bar(confidence))

    converge_text = Text()
    if converging:
        converge_text.append("YES", style="bold green")
        converge_text.append(" — multiple methods agree")
    else:
        converge_text.append("NO", style="dim")
        converge_text.append(" — single-method findings only")
    table.add_row("Converging Evidence", converge_text)

    findings_text = Text(str(total_findings), style="bold")
    if critical_count > 0:
        findings_text.append(f" ({critical_count} critical)", style="bold red")
    table.add_row("Total Findings", findings_text)

    console.print(table)
    console.print()


def print_findings_table(findings: list[dict]) -> None:
    """Print findings sorted by severity with colored badges."""
    if not findings:
        console.print("[dim]No findings to report.[/dim]\n")
        return

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    sorted_findings = sorted(
        findings, key=lambda f: severity_order.get(f.get("severity", "info"), 5)
    )

    table = Table(title="Findings", border_style="bright_blue")
    table.add_column("#", style="dim", width=3)
    table.add_column("Severity", width=12)
    table.add_column("Method", style="cyan", width=18)
    table.add_column("Title", width=40)
    table.add_column("Confidence", width=10)
    table.add_column("Figure", style="dim", width=12)

    for i, finding in enumerate(sorted_findings, 1):
        severity = finding.get("severity", "info")
        table.add_row(
            str(i),
            _severity_badge(severity),
            finding.get("analysis_type", finding.get("method", "unknown")),
            finding.get("title", "Untitled"),
            f"{finding.get('confidence', 0.0):.0%}",
            finding.get("figure_label", finding.get("figure_id", "")),
        )

    console.print(table)
    console.print()


def print_methods_summary(methods: list[dict]) -> None:
    """Print a summary table of analysis methods applied."""
    if not methods:
        return

    table = Table(title="Methods Applied", border_style="bright_blue")
    table.add_column("Analysis Type", style="cyan", width=24)
    table.add_column("Figures Checked", justify="center", width=16)
    table.add_column("Issues Found", justify="center", width=14)

    for method in methods:
        issues = method.get("issues_found", 0)
        issues_style = "bold red" if issues > 0 else "green"
        table.add_row(
            method["name"],
            str(method.get("figures_analyzed", 0)),
            Text(str(issues), style=issues_style),
        )

    console.print(table)
    console.print()


def print_figure_detail(figure_id: str, findings: list[dict]) -> None:
    """Print a detail panel for a single figure's findings."""
    content = Text()
    for finding in findings:
        sev = finding.get("severity", "info")
        content.append(f"  [{sev.upper()}] ", style=SEVERITY_STYLES.get(sev, "dim"))
        content.append(f"{finding.get('title', 'Untitled')}\n", style="white")
        desc = finding.get("description", "")
        if desc:
            content.append(f"    {desc}\n", style="dim")

        # Show key evidence details
        evidence = finding.get("evidence", {})
        if isinstance(evidence, dict):
            for key, val in evidence.items():
                content.append(f"    {key}: ", style="dim cyan")
                content.append(f"{val}\n", style="white")
        content.append("\n")

    console.print(Panel(content, title=f"[bold]{figure_id}[/bold]", border_style="dim"))


def print_full_report(
    paper: dict,
    findings: list[dict],
    figures: list[dict] | None = None,
    evidence: dict | None = None,
) -> None:
    """Orchestrate the full pretty report output.

    Args:
        paper: Paper metadata dict.
        findings: List of finding dicts.
        figures: Optional list of figure info dicts.
        evidence: Optional aggregated evidence dict with keys like
            ``overall_risk``, ``overall_confidence``, ``converging_evidence``,
            ``total_findings``, ``critical_count``, ``methods_summary``.
    """
    console.print()
    console.rule("[bold blue]Snoopy Analysis Report[/bold blue]")
    console.print()

    print_paper_header(paper)
    console.print()

    # Assessment section
    if evidence:
        print_assessment(
            risk=evidence.get("overall_risk", "clean"),
            confidence=evidence.get("overall_confidence", 0.0),
            converging=evidence.get("converging_evidence", False),
            total_findings=evidence.get("total_findings", len(findings)),
            critical_count=evidence.get("critical_count", 0),
        )
    else:
        # Compute basic stats from findings
        critical = sum(1 for f in findings if f.get("severity") == "critical")
        has_high = any(f.get("severity") in ("critical", "high") for f in findings)
        risk = (
            "high" if critical > 0 else ("medium" if has_high else ("low" if findings else "clean"))
        )
        print_assessment(
            risk=risk,
            confidence=0.5 if findings else 0.0,
            converging=False,
            total_findings=len(findings),
            critical_count=critical,
        )

    # Findings table
    print_findings_table(findings)

    # Methods summary
    if evidence and evidence.get("methods_summary"):
        print_methods_summary(evidence["methods_summary"])

    # Per-figure detail panels (only for figures with findings)
    if findings:
        figure_groups: dict[str, list[dict]] = {}
        for f in findings:
            fig_id = f.get("figure_id", f.get("figure_label", "unknown"))
            figure_groups.setdefault(fig_id, []).append(f)

        if any(len(v) > 0 for v in figure_groups.values()):
            console.rule("[dim]Figure Details[/dim]")
            console.print()
            for fig_id, fig_findings in figure_groups.items():
                print_figure_detail(fig_id, fig_findings)

    console.rule("[bold blue]End of Report[/bold blue]")
    console.print()


def print_demo_summary(results: list[dict]) -> None:
    """Print a summary dashboard table for demo results.

    Args:
        results: List of dicts with keys: name, category, expected, actual_risk,
            findings_count, pass_fail.
    """
    console.print()
    console.rule("[bold green]Demo Summary Dashboard[/bold green]")
    console.print()

    table = Table(title="Test Results", border_style="green")
    table.add_column("#", style="dim", width=3)
    table.add_column("Test Case", width=35)
    table.add_column("Category", style="cyan", width=18)
    table.add_column("Expected", width=12)
    table.add_column("Risk Found", width=12)
    table.add_column("Findings", justify="center", width=10)
    table.add_column("Result", width=8)

    passed = 0
    failed = 0
    for i, r in enumerate(results, 1):
        pass_fail = r.get("pass_fail", False)
        if pass_fail:
            passed += 1
            result_text = Text(" PASS ", style="bold white on green")
        else:
            failed += 1
            result_text = Text(" FAIL ", style="bold white on red")

        risk = r.get("actual_risk", "clean")
        table.add_row(
            str(i),
            r.get("name", "Unknown"),
            r.get("category", ""),
            r.get("expected", ""),
            _risk_badge(risk),
            str(r.get("findings_count", 0)),
            result_text,
        )

    console.print(table)
    console.print()

    total = passed + failed
    summary = Text()
    summary.append(
        f"  {passed}/{total} passed", style="bold green" if failed == 0 else "bold yellow"
    )
    if failed > 0:
        summary.append(f"  |  {failed} failed", style="bold red")
    console.print(Panel(summary, title="[bold]Summary[/bold]", border_style="green"))
    console.print()


def create_progress() -> Progress:
    """Create a rich Progress bar for long-running operations."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    )
