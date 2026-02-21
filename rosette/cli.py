"""Click-based CLI entry point for rosette."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

import click

from rosette.config import load_config
from rosette.db.session import get_session, init_async_db, init_db


def setup_logging(verbose: bool, json_logs: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    if json_logs:
        import json as _json

        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_entry: dict = {
                    "timestamp": self.formatTime(record, self.datefmt),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                # Include structured context fields if present
                for field in ("paper_id", "stage", "figure_id"):
                    val = getattr(record, field, None)
                    if val is not None:
                        log_entry[field] = val
                return _json.dumps(log_entry)

        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
        logging.root.handlers.clear()
        logging.root.addHandler(handler)
        logging.root.setLevel(level)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


@click.group()
@click.option("--config", "-c", "config_path", default=None, help="Path to config YAML file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--json-logs", is_flag=True, help="Output logs in JSON format")
@click.pass_context
def main(ctx: click.Context, config_path: str | None, verbose: bool, json_logs: bool) -> None:
    """Rosette: LLM-powered academic integrity analyzer."""
    setup_logging(verbose, json_logs=json_logs)
    ctx.ensure_object(dict)
    cfg = load_config(config_path)
    ctx.obj["config"] = cfg
    init_db(cfg.storage.database_url)
    init_async_db(cfg.storage.database_url)


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
        from rosette.db.models import Paper
        from rosette.discovery.openalex import search_works
        from rosette.discovery.priority import PaperMetadata, compute_priority

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
                doi = work.doi
                if doi:
                    from sqlalchemy import select

                    existing = (
                        session.execute(select(Paper).where(Paper.doi == doi)).scalars().first()
                    )
                    if existing:
                        continue

                metadata = PaperMetadata(
                    citation_count=work.citation_count,
                    journal_quartile=getattr(work, "journal_quartile", 2),
                    influential_citations=getattr(work, "influential_citation_count", 0),
                    max_author_hindex=getattr(work, "max_author_hindex", 0),
                    institution_in_top100=getattr(work, "institution_in_top100", False),
                    has_retraction_concern=getattr(work, "has_retraction_concern", False),
                    year=work.publication_year or 2020,
                    has_image_heavy_methods=getattr(work, "has_image_heavy_methods", False),
                )
                priority = compute_priority(metadata)

                min_priority = config.priority.min_priority_score
                if priority < min_priority:
                    continue

                paper = Paper(
                    id=str(uuid.uuid4()),
                    doi=doi,
                    title=work.title or "Unknown",
                    abstract=work.abstract,
                    authors_json=json.dumps([a.name for a in work.authors]),
                    journal=work.journal,
                    journal_issn=work.issn,
                    publication_year=work.publication_year,
                    citation_count=work.citation_count,
                    influential_citation_count=getattr(work, "influential_citation_count", None),
                    priority_score=priority,
                    source="openalex",
                    status="pending",
                )
                session.add(paper)
                added += 1

        click.echo(f"Added {added} papers to queue.")

    asyncio.run(_run())


def _validate_doi(doi: str) -> str:
    """Validate and normalize a DOI string."""
    from rosette.validation import validate_doi

    try:
        return validate_doi(doi)
    except ValueError as e:
        raise click.BadParameter(str(e))


@main.command()
@click.option("--doi", default=None, help="DOI of paper to analyze")
@click.option("--pdf", "pdf_path", default=None, type=click.Path(exists=True), help="Path to PDF")
@click.option("--from-stage", default=None, help="Start from this pipeline stage")
@click.option("--to-stage", default=None, help="Stop after this pipeline stage")
@click.option("--force-stage", multiple=True, help="Force re-run of specific stage(s)")
@click.pass_context
def analyze(
    ctx: click.Context,
    doi: str | None,
    pdf_path: str | None,
    from_stage: str | None,
    to_stage: str | None,
    force_stage: tuple[str, ...],
) -> None:
    """Analyze a single paper by DOI or local PDF path."""
    config = ctx.obj["config"]

    if not doi and not pdf_path:
        click.echo("Error: provide either --doi or --pdf", err=True)
        sys.exit(1)

    if doi:
        doi = _validate_doi(doi)

    async def _run() -> None:
        from rosette.db.models import Paper
        from rosette.pipeline.orchestrator import PipelineOrchestrator

        paper_id = str(uuid.uuid4())

        with get_session() as session:
            if doi:
                from sqlalchemy import select

                existing = session.execute(select(Paper).where(Paper.doi == doi)).scalars().first()
                if existing:
                    paper_id = str(existing.id)
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

                pdf = Path(pdf_path)  # type: ignore[arg-type]
                h = hashlib.sha256()
                with open(pdf, "rb") as fh:
                    for chunk in iter(lambda: fh.read(8192), b""):
                        h.update(chunk)
                sha256 = h.hexdigest()
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
        if force_stage or from_stage or to_stage:
            await orchestrator.process_paper_stages(
                paper_id,
                from_stage=from_stage,
                to_stage=to_stage,
                force_stages=list(force_stage) if force_stage else None,
            )
        else:
            await orchestrator.process_paper(paper_id)
        click.echo(f"Analysis complete. Paper ID: {paper_id}")
        click.echo(f"Run 'rosette report --paper-id {paper_id}' to view the report.")

    asyncio.run(_run())


@main.command()
@click.option("--limit", default=50, help="Number of papers to process")
@click.option("--min-priority", default=60.0, help="Minimum priority score")
@click.pass_context
def batch(ctx: click.Context, limit: int, min_priority: float) -> None:
    """Process top-priority pending papers in batch."""
    config = ctx.obj["config"]

    async def _run() -> None:
        from rosette.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(config)
        click.echo(f"Processing up to {limit} papers with priority >= {min_priority}...")
        results = await orchestrator.run_batch(limit=limit, min_priority=min_priority)
        click.echo(f"Processed {len(results)} papers.")

    asyncio.run(_run())


@main.command()
@click.option("--paper-id", required=True, help="Paper UUID")
@click.option("--format", "fmt", type=click.Choice(["markdown", "html"]), default="markdown")
@click.pass_context
def report(ctx: click.Context, paper_id: str, fmt: str) -> None:
    """Display or regenerate a paper's analysis report."""
    from sqlalchemy import select

    from rosette.db.models import Report

    with get_session() as session:
        rpt = (
            session.execute(
                select(Report).where(Report.paper_id == paper_id).order_by(Report.created_at.desc())
            )
            .scalars()
            .first()
        )

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

    from rosette.db.models import Paper

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
        top = (
            session.execute(
                select(Paper)
                .where(Paper.status == "pending")
                .order_by(Paper.priority_score.desc())
                .limit(5)
            )
            .scalars()
            .all()
        )

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
@click.option(
    "--seed",
    default=42,
    type=int,
    show_default=True,
    help="Random seed for RSIIL sample selection",
)
@click.option(
    "--sample-size",
    "-M",
    default=50,
    type=int,
    show_default=True,
    help="Number of RSIIL images to sample per category",
)
@click.pass_context
def demo(
    ctx: click.Context,
    download_only: bool,
    skip_llm: bool,
    output_dir: str | None,
    download_rsiil: bool,
    seed: int,
    sample_size: int,
) -> None:
    """Run the demo pipeline with test fixtures and pretty output."""
    from rosette.demo.runner import run_demo

    run_demo(
        download_only=download_only,
        skip_llm=skip_llm,
        output_dir=output_dir,
        download_rsiil=download_rsiil,
        seed=seed,
        sample_size=sample_size,
    )


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")  # nosec B104
@click.option("--port", default=8000, help="Port to bind to")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int) -> None:
    """Start the API server."""
    import uvicorn

    from rosette.api.app import create_app

    app = create_app(ctx.obj["config"])
    uvicorn.run(app, host=host, port=port)


@main.command("config")
@click.pass_context
def show_config(ctx: click.Context) -> None:
    """Show current configuration."""
    config = ctx.obj["config"]
    click.echo(config.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# db subcommand group  --  Alembic migration helpers
# ---------------------------------------------------------------------------


@main.group()
@click.pass_context
def db(ctx: click.Context) -> None:
    """Database migration commands (powered by Alembic)."""


@db.command()
@click.option(
    "--revision",
    default="head",
    show_default=True,
    help="Revision target (default: head).",
)
@click.pass_context
def upgrade(ctx: click.Context, revision: str) -> None:
    """Run migrations forward to the target revision (default: head)."""
    from alembic import command as alembic_cmd

    cfg = ctx.obj["config"]
    alembic_cfg = _make_alembic_config(cfg.storage.database_url)
    click.echo(f"Upgrading database to revision '{revision}' ...")
    alembic_cmd.upgrade(alembic_cfg, revision)
    click.echo("Done.")


@db.command()
@click.option(
    "--revision",
    default="-1",
    show_default=True,
    help="Revision target (default: -1, i.e. rollback one step).",
)
@click.pass_context
def downgrade(ctx: click.Context, revision: str) -> None:
    """Rollback migrations to the target revision (default: one step back)."""
    from alembic import command as alembic_cmd

    cfg = ctx.obj["config"]
    alembic_cfg = _make_alembic_config(cfg.storage.database_url)
    click.echo(f"Downgrading database to revision '{revision}' ...")
    alembic_cmd.downgrade(alembic_cfg, revision)
    click.echo("Done.")


@db.command()
@click.pass_context
def current(ctx: click.Context) -> None:
    """Show the current migration revision stamped in the database."""
    from alembic import command as alembic_cmd

    cfg = ctx.obj["config"]
    alembic_cfg = _make_alembic_config(cfg.storage.database_url)
    alembic_cmd.current(alembic_cfg, verbose=True)


def _make_alembic_config(database_url: str):
    """Build an :class:`alembic.config.Config` pointing at the project ini."""
    from pathlib import Path as _Path

    from alembic.config import Config as AlembicConfig

    ini_path = _Path(__file__).resolve().parent.parent / "alembic.ini"
    alembic_cfg = AlembicConfig(str(ini_path))
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    return alembic_cfg


# ---------------------------------------------------------------------------
# campaign subcommand group
# ---------------------------------------------------------------------------

from rosette.cli_campaign import campaign  # noqa: E402

main.add_command(campaign)


if __name__ == "__main__":
    main()
