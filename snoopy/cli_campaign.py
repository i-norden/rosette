"""CLI commands for campaign investigation system."""

from __future__ import annotations

import asyncio
import json
import uuid

import click

from snoopy.db.session import get_session


@click.group()
def campaign() -> None:
    """Campaign investigation commands."""


@campaign.command("create")
@click.option(
    "--mode",
    type=click.Choice(["network_expansion", "domain_scan", "paper_mill"]),
    required=True,
    help="Campaign investigation mode",
)
@click.option("--name", required=True, help="Human-readable campaign name")
@click.option("--seed-doi", "seed_dois", multiple=True, help="Seed DOI (repeatable)")
@click.option("--field", "research_field", default=None, help="Research field for domain scan")
@click.option("--journal", default=None, help="Journal filter for domain scan")
@click.option("--min-citations", default=50, help="Minimum citation count")
@click.option("--max-depth", default=2, help="Network expansion depth limit")
@click.option("--max-papers", default=1000, help="Maximum papers to discover")
@click.option("--llm-budget", default=100, help="Max papers for LLM analysis")
@click.pass_context
def create(
    ctx: click.Context,
    mode: str,
    name: str,
    seed_dois: tuple[str, ...],
    research_field: str | None,
    journal: str | None,
    min_citations: int,
    max_depth: int,
    max_papers: int,
    llm_budget: int,
) -> None:
    """Create a new campaign investigation."""
    from snoopy.db.models import Campaign

    # Validate mode-specific requirements
    if mode in ("network_expansion", "paper_mill") and not seed_dois:
        click.echo(
            "Error: --seed-doi is required for network_expansion and paper_mill modes", err=True
        )
        raise SystemExit(1)

    if mode == "domain_scan" and not research_field:
        click.echo("Error: --field is required for domain_scan mode", err=True)
        raise SystemExit(1)

    # Build config JSON
    config_data: dict = {
        "min_citations": min_citations,
    }
    if research_field:
        config_data["field"] = research_field
    if journal:
        config_data["journal"] = journal

    campaign_id = str(uuid.uuid4())

    with get_session() as session:
        c = Campaign(
            id=campaign_id,
            name=name,
            mode=mode,
            status="created",
            config_json=json.dumps(config_data),
            seed_dois=json.dumps(list(seed_dois)) if seed_dois else None,
            max_depth=max_depth,
            max_papers=max_papers,
            llm_budget=llm_budget,
        )
        session.add(c)

    click.echo(f"Campaign created: {campaign_id}")
    click.echo(f"  Name: {name}")
    click.echo(f"  Mode: {mode}")
    if seed_dois:
        click.echo(f"  Seeds: {len(seed_dois)} DOIs")
    click.echo(f"  Max papers: {max_papers}, LLM budget: {llm_budget}")
    click.echo(f"\nRun with: snoopy campaign run {campaign_id}")


@campaign.command("run")
@click.argument("campaign_id")
@click.pass_context
def run_campaign(ctx: click.Context, campaign_id: str) -> None:
    """Start or resume a campaign investigation."""
    config = ctx.obj["config"]

    async def _run() -> None:
        from snoopy.campaign.orchestrator import CampaignOrchestrator

        orchestrator = CampaignOrchestrator(config, campaign_id)
        click.echo(f"Starting campaign {campaign_id}...")
        await orchestrator.run()
        click.echo("Campaign completed.")

    asyncio.run(_run())


@campaign.command("pause")
@click.argument("campaign_id")
@click.pass_context
def pause_campaign(ctx: click.Context, campaign_id: str) -> None:
    """Pause a running campaign (active tasks will complete)."""
    from snoopy.db.models import Campaign

    with get_session() as session:
        c = session.get(Campaign, campaign_id)
        if not c:
            click.echo(f"Campaign {campaign_id} not found", err=True)
            raise SystemExit(1)
        c.status = "paused"  # type: ignore[assignment]

    click.echo(f"Campaign {campaign_id} paused.")


@campaign.command("status")
@click.argument("campaign_id", required=False)
@click.pass_context
def campaign_status(ctx: click.Context, campaign_id: str | None) -> None:
    """Show campaign progress summary."""
    from sqlalchemy import select

    from snoopy.db.models import Campaign

    with get_session() as session:
        if campaign_id:
            c = session.get(Campaign, campaign_id)
            if not c:
                click.echo(f"Campaign {campaign_id} not found", err=True)
                raise SystemExit(1)
            _print_campaign_detail(session, c)
        else:
            campaigns = (
                session.execute(select(Campaign).order_by(Campaign.created_at.desc()))
                .scalars()
                .all()
            )
            if not campaigns:
                click.echo("No campaigns found.")
                return
            for c in campaigns:
                _print_campaign_summary(c)
                click.echo()


@campaign.command("list")
@click.pass_context
def list_campaigns(ctx: click.Context) -> None:
    """List all campaigns."""
    from sqlalchemy import select

    from snoopy.db.models import Campaign

    with get_session() as session:
        campaigns = (
            session.execute(select(Campaign).order_by(Campaign.created_at.desc())).scalars().all()
        )

        if not campaigns:
            click.echo("No campaigns found.")
            return

        click.echo(f"{'ID':<38} {'Name':<30} {'Mode':<20} {'Status':<15} {'Papers':<8}")
        click.echo("-" * 111)
        for c in campaigns:
            click.echo(
                f"{c.id:<38} {str(c.name)[:29]:<30} {str(c.mode):<20} "
                f"{str(c.status):<15} {c.papers_discovered or 0:<8}"
            )


@campaign.command("dashboard")
@click.argument("campaign_id")
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Output HTML file path",
)
@click.pass_context
def dashboard(ctx: click.Context, campaign_id: str, output: str | None) -> None:
    """Generate an HTML dashboard for a campaign."""
    config = ctx.obj["config"]

    from snoopy.campaign.dashboard import generate_campaign_dashboard
    from snoopy.db.models import Campaign

    with get_session() as session:
        c = session.get(Campaign, campaign_id)
        if not c:
            click.echo(f"Campaign {campaign_id} not found", err=True)
            raise SystemExit(1)

    html = generate_campaign_dashboard(campaign_id)

    if output:
        from pathlib import Path

        Path(output).write_text(html)
        click.echo(f"Dashboard written to {output}")
    else:
        # Default output path
        from pathlib import Path

        reports_dir = Path(config.storage.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir / f"campaign_{campaign_id[:8]}_dashboard.html"
        out_path.write_text(html)
        click.echo(f"Dashboard written to {out_path}")


@campaign.command("export")
@click.argument("campaign_id")
@click.option(
    "--min-risk",
    type=click.Choice(["critical", "high", "medium", "low"]),
    default="medium",
    help="Minimum risk level to include",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(),
    help="Output directory for evidence packages",
)
@click.pass_context
def export_campaign(
    ctx: click.Context, campaign_id: str, min_risk: str, output_dir: str | None
) -> None:
    """Export evidence packages for flagged papers."""
    from pathlib import Path

    from sqlalchemy import select

    from snoopy.db.models import Campaign, CampaignPaper, Paper, Report

    config = ctx.obj["config"]
    risk_levels = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    min_level = risk_levels.get(min_risk, 2)

    with get_session() as session:
        c = session.get(Campaign, campaign_id)
        if not c:
            click.echo(f"Campaign {campaign_id} not found", err=True)
            raise SystemExit(1)

        # Get campaign papers meeting risk threshold
        campaign_papers = (
            session.execute(
                select(CampaignPaper)
                .where(CampaignPaper.campaign_id == campaign_id)
                .where(CampaignPaper.final_risk.isnot(None))
            )
            .scalars()
            .all()
        )

        export_dir = (
            Path(output_dir)
            if output_dir
            else Path(config.storage.reports_dir) / f"campaign_{campaign_id[:8]}_export"
        )
        export_dir.mkdir(parents=True, exist_ok=True)

        exported = 0
        for cp in campaign_papers:
            final_risk = str(cp.final_risk)
            if risk_levels.get(final_risk, 4) > min_level:
                continue

            paper = session.get(Paper, str(cp.paper_id))
            if not paper:
                continue

            report = (
                session.execute(
                    select(Report)
                    .where(Report.paper_id == str(cp.paper_id))
                    .order_by(Report.created_at.desc())
                )
                .scalars()
                .first()
            )

            if report and report.report_html:
                out_file = export_dir / f"{cp.paper_id}.html"
                out_file.write_text(str(report.report_html))
                exported += 1

        click.echo(f"Exported {exported} evidence packages to {export_dir}")


def _print_campaign_summary(c) -> None:  # type: ignore[no-untyped-def]
    """Print a one-line campaign summary."""
    click.echo(
        f"[{c.status}] {c.name} ({c.mode}) - "
        f"{c.papers_discovered or 0} discovered, "
        f"{c.papers_triaged or 0} triaged, "
        f"{c.papers_flagged or 0} flagged, "
        f"{c.papers_llm_analyzed or 0} LLM analyzed"
    )
    click.echo(f"  ID: {c.id}")


def _print_campaign_detail(session, c) -> None:  # type: ignore[no-untyped-def]
    """Print detailed campaign status."""
    from sqlalchemy import func, select

    from snoopy.db.models import CampaignPaper

    click.echo(f"Campaign: {c.name}")
    click.echo(f"  ID:     {c.id}")
    click.echo(f"  Mode:   {c.mode}")
    click.echo(f"  Status: {c.status}")
    click.echo(f"  Created: {c.created_at}")
    click.echo(f"  Updated: {c.updated_at}")
    click.echo()
    click.echo("Progress:")
    click.echo(f"  Papers discovered:   {c.papers_discovered or 0}")
    click.echo(f"  Papers triaged:      {c.papers_triaged or 0}")
    click.echo(f"  Papers flagged:      {c.papers_flagged or 0}")
    click.echo(f"  Papers LLM analyzed: {c.papers_llm_analyzed or 0} / {c.llm_budget or 100}")
    click.echo()

    # Triage status breakdown
    triage_counts = session.execute(
        select(CampaignPaper.triage_status, func.count(CampaignPaper.id))
        .where(CampaignPaper.campaign_id == str(c.id))
        .group_by(CampaignPaper.triage_status)
    ).all()

    if triage_counts:
        click.echo("Triage breakdown:")
        for status_val, count in triage_counts:
            click.echo(f"  {status_val}: {count}")
        click.echo()

    # Risk distribution
    risk_counts = session.execute(
        select(CampaignPaper.final_risk, func.count(CampaignPaper.id))
        .where(CampaignPaper.campaign_id == str(c.id))
        .where(CampaignPaper.final_risk.isnot(None))
        .group_by(CampaignPaper.final_risk)
    ).all()

    if risk_counts:
        click.echo("Risk distribution:")
        for risk_val, count in risk_counts:
            click.echo(f"  {risk_val}: {count}")
