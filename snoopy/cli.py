"""Click-based CLI entry point for snoopy."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

import click

from snoopy.config import load_config
from snoopy.db.session import get_session, init_db


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.option("--config", "-c", "config_path", default=None, help="Path to config YAML file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx: click.Context, config_path: str | None, verbose: bool) -> None:
    """Snoopy: LLM-powered academic integrity analyzer."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    cfg = load_config(config_path)
    ctx.obj["config"] = cfg
    init_db(cfg.storage.database_url)


@main.command()
@click.option("--field", "research_field", default="biomedical", help="Research field to search")
@click.option("--min-citations", default=100, help="Minimum citation count")
@click.option("--journal-quartile", default=None, help="Journal quartile filter (Q1, Q2, etc.)")
@click.option("--limit", default=500, help="Maximum papers to discover")
@click.pass_context
def discover(
    ctx: click.Context,
    research_field: str,
    min_citations: int,
    journal_quartile: str | None,
    limit: int,
) -> None:
    """Discover papers from academic APIs and compute priority scores."""
    config = ctx.obj["config"]

    async def _run() -> None:
        from snoopy.db.models import Paper
        from snoopy.discovery.openalex import search_works
        from snoopy.discovery.priority import PaperMetadata, compute_priority

        click.echo(f"Searching for {research_field} papers with >= {min_citations} citations...")

        works = await search_works(
            query=research_field,
            field=research_field,
            min_citations=min_citations,
            limit=limit,
        )

        click.echo(f"Found {len(works)} papers. Computing priority scores...")

        added = 0
        with get_session() as session:
            for work in works:
                doi = work.get("doi")
                if doi:
                    from sqlalchemy import select
                    existing = session.execute(
                        select(Paper).where(Paper.doi == doi)
                    ).scalars().first()
                    if existing:
                        continue

                metadata = PaperMetadata(
                    citation_count=work.get("citation_count", 0),
                    journal_quartile=work.get("journal_quartile", 2),
                    influential_citations=work.get("influential_citation_count", 0),
                    max_author_hindex=work.get("max_author_hindex", 0),
                    institution_in_top100=work.get("institution_in_top100", False),
                    has_retraction_concern=work.get("has_retraction_concern", False),
                    year=work.get("publication_year", 2020),
                    has_image_heavy_methods=work.get("has_image_heavy_methods", False),
                )
                priority = compute_priority(metadata)

                min_priority = config.priority.min_priority_score
                if priority < min_priority:
                    continue

                paper = Paper(
                    id=str(uuid.uuid4()),
                    doi=doi,
                    title=work.get("title", "Unknown"),
                    abstract=work.get("abstract"),
                    authors_json=json.dumps(work.get("authors", [])),
                    journal=work.get("journal"),
                    journal_issn=work.get("issn"),
                    publication_year=work.get("publication_year"),
                    citation_count=work.get("citation_count"),
                    influential_citation_count=work.get("influential_citation_count"),
                    priority_score=priority,
                    source="openalex",
                    status="pending",
                )
                session.add(paper)
                added += 1

        click.echo(f"Added {added} papers to queue.")

    asyncio.run(_run())


@main.command()
@click.option("--doi", default=None, help="DOI of paper to analyze")
@click.option("--pdf", "pdf_path", default=None, type=click.Path(exists=True), help="Path to PDF")
@click.pass_context
def analyze(ctx: click.Context, doi: str | None, pdf_path: str | None) -> None:
    """Analyze a single paper by DOI or local PDF path."""
    config = ctx.obj["config"]

    if not doi and not pdf_path:
        click.echo("Error: provide either --doi or --pdf", err=True)
        sys.exit(1)

    async def _run() -> None:
        from snoopy.db.models import Paper
        from snoopy.pipeline.orchestrator import PipelineOrchestrator

        paper_id = str(uuid.uuid4())

        with get_session() as session:
            if doi:
                from sqlalchemy import select
                existing = session.execute(
                    select(Paper).where(Paper.doi == doi)
                ).scalars().first()
                if existing:
                    paper_id = existing.id
                    click.echo(f"Paper already in database: {paper_id}")
                else:
                    paper = Paper(
                        id=paper_id,
                        doi=doi,
                        title=doi,  # Will be updated during processing
                        source="manual",
                        status="pending",
                    )
                    session.add(paper)
                    click.echo(f"Added paper {paper_id} for DOI {doi}")
            else:
                import hashlib
                pdf = Path(pdf_path)
                sha256 = hashlib.sha256(pdf.read_bytes()).hexdigest()
                paper = Paper(
                    id=paper_id,
                    title=pdf.stem,
                    pdf_path=str(pdf.resolve()),
                    pdf_sha256=sha256,
                    source="manual",
                    status="pending",
                )
                session.add(paper)
                click.echo(f"Added paper {paper_id} from {pdf_path}")

        orchestrator = PipelineOrchestrator(config)
        click.echo("Starting analysis pipeline...")
        await orchestrator.process_paper(paper_id)
        click.echo(f"Analysis complete. Paper ID: {paper_id}")
        click.echo(f"Run 'snoopy report --paper-id {paper_id}' to view the report.")

    asyncio.run(_run())


@main.command()
@click.option("--limit", default=50, help="Number of papers to process")
@click.option("--min-priority", default=60.0, help="Minimum priority score")
@click.pass_context
def batch(ctx: click.Context, limit: int, min_priority: float) -> None:
    """Process top-priority pending papers in batch."""
    config = ctx.obj["config"]

    async def _run() -> None:
        from snoopy.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(config)
        click.echo(f"Processing up to {limit} papers with priority >= {min_priority}...")
        results = await orchestrator.run_batch(limit=limit)
        click.echo(f"Processed {len(results)} papers.")

    asyncio.run(_run())


@main.command()
@click.option("--paper-id", required=True, help="Paper UUID")
@click.option("--format", "fmt", type=click.Choice(["markdown", "html"]), default="markdown")
@click.pass_context
def report(ctx: click.Context, paper_id: str, fmt: str) -> None:
    """Display or regenerate a paper's analysis report."""
    from sqlalchemy import select

    from snoopy.db.models import Report

    with get_session() as session:
        rpt = session.execute(
            select(Report)
            .where(Report.paper_id == paper_id)
            .order_by(Report.created_at.desc())
        ).scalars().first()

        if not rpt:
            click.echo(f"No report found for paper {paper_id}", err=True)
            sys.exit(1)

        if fmt == "markdown":
            click.echo(rpt.report_markdown)
        else:
            click.echo(rpt.report_html)


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show pipeline status and queue depth."""
    from sqlalchemy import func, select

    from snoopy.db.models import Paper

    with get_session() as session:
        total = session.execute(select(func.count(Paper.id))).scalar() or 0
        results = session.execute(
            select(Paper.status, func.count(Paper.id)).group_by(Paper.status)
        ).all()

        click.echo(f"Total papers: {total}")
        click.echo("Status breakdown:")
        for status_val, count in results:
            click.echo(f"  {status_val}: {count}")

        # Show top priority pending
        top = session.execute(
            select(Paper)
            .where(Paper.status == "pending")
            .order_by(Paper.priority_score.desc())
            .limit(5)
        ).scalars().all()

        if top:
            click.echo("\nTop priority pending papers:")
            for p in top:
                click.echo(
                    f"  [{p.priority_score or 0:.1f}] {p.doi or 'no-doi'} - "
                    f"{(p.title or 'Untitled')[:60]}"
                )


@main.command()
@click.option("--download-only", is_flag=True, help="Only download fixtures, don't run analysis")
@click.option("--skip-llm/--use-llm", default=True, help="Skip LLM-based analysis (default: skip)")
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(),
    help="Output directory for HTML reports",
)
@click.option(
    "--download-rsiil",
    is_flag=True,
    help="Download full RSIIL dataset from Zenodo (~57 GB)",
)
@click.pass_context
def demo(
    ctx: click.Context,
    download_only: bool,
    skip_llm: bool,
    output_dir: str | None,
    download_rsiil: bool,
) -> None:
    """Run the demo pipeline with test fixtures and pretty output."""
    from snoopy.demo.runner import run_demo

    run_demo(
        download_only=download_only,
        skip_llm=skip_llm,
        output_dir=output_dir,
        download_rsiil=download_rsiil,
    )


@main.command("config")
@click.pass_context
def show_config(ctx: click.Context) -> None:
    """Show current configuration."""
    config = ctx.obj["config"]
    click.echo(config.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
