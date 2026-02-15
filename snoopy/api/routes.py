"""API route handlers for the snoopy FastAPI application."""

from __future__ import annotations

import base64
import binascii
import hmac
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from snoopy.api.schemas import (
    AuthorRiskResponse,
    BatchStatusResponse,
    BatchSubmitRequest,
    FindingResponse,
    PaperStatusResponse,
    PaperSubmitRequest,
    ReportResponse,
)
from snoopy.db.models import Author, Finding, Paper, Report
from snoopy.db.session import get_async_session
from snoopy.validation import validate_doi

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


async def _verify_api_key(request: Request) -> None:
    """Dependency that validates the X-API-Key header against configured keys."""
    config = request.app.state.config
    api_keys = getattr(config, "api_keys", None)
    if not api_keys:
        require_auth = getattr(config, "require_authentication", True)
        if require_auth:
            raise HTTPException(
                status_code=500,
                detail="Server misconfigured: API keys are required but none are configured. "
                "Set api_keys in config or set require_authentication: false for dev mode.",
            )
        logger.warning(
            "SECURITY: No API keys configured - all requests allowed (require_authentication=false)"
        )
        return
    key = request.headers.get("X-API-Key")
    if not key or not any(hmac.compare_digest(key, k) for k in api_keys):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def _run_pipeline(request: Request, paper_id: str) -> None:
    """Background task to run the analysis pipeline on a paper."""
    orchestrator = request.app.state.orchestrator
    if orchestrator is None:
        logger.error("Orchestrator not initialized, cannot process paper %s", paper_id)
        return
    try:
        await orchestrator.process_paper(paper_id)
    except Exception:
        logger.exception("Pipeline failed for paper %s", paper_id)


@router.post("/papers", status_code=202, response_model=PaperStatusResponse)
async def submit_paper(
    request: Request,
    body: PaperSubmitRequest,
    background_tasks: BackgroundTasks,
    _auth: None = Depends(_verify_api_key),
) -> PaperStatusResponse:
    """Submit a paper for analysis by DOI or PDF upload.

    Returns 202 Accepted with the paper_id for tracking.
    """
    if not body.doi and not body.pdf_upload:
        raise HTTPException(status_code=422, detail="Provide either a DOI or PDF upload")

    if body.pdf_upload:
        try:
            pdf_bytes = base64.b64decode(body.pdf_upload, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=422, detail="Invalid base64 encoding for pdf_upload")
        if not pdf_bytes[:5].startswith(b"%PDF-"):
            raise HTTPException(status_code=422, detail="Uploaded content is not a valid PDF")

    if body.doi:
        try:
            body.doi = validate_doi(body.doi)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    paper_id = str(uuid.uuid4())

    try:
        async with get_async_session() as session:
            if body.doi:
                existing = (
                    (await session.execute(select(Paper).where(Paper.doi == body.doi)))
                    .scalars()
                    .first()
                )
                if existing:
                    return PaperStatusResponse(
                        paper_id=str(existing.id),
                        status=str(existing.status),
                        risk_level=None,
                        overall_confidence=None,
                        num_findings=None,
                    )

                paper = Paper(
                    id=paper_id,
                    doi=body.doi,
                    title=body.doi,
                    source="api",
                    status="pending",
                )
                session.add(paper)
            else:
                paper = Paper(
                    id=paper_id,
                    title="uploaded_pdf",
                    source="api_upload",
                    status="pending",
                )
                session.add(paper)

        # Only start pipeline for newly created papers (not duplicates returned above)
        background_tasks.add_task(_run_pipeline, request, paper_id)
    except IntegrityError:
        # Race condition: another request inserted the same DOI concurrently
        if not body.doi:
            raise HTTPException(status_code=500, detail="Unexpected database error")
        async with get_async_session() as session:
            existing = (
                (await session.execute(select(Paper).where(Paper.doi == body.doi)))
                .scalars()
                .first()
            )
            if existing:
                return PaperStatusResponse(
                    paper_id=str(existing.id),
                    status=str(existing.status),
                    risk_level=None,
                    overall_confidence=None,
                    num_findings=None,
                )
            raise HTTPException(status_code=500, detail="Unexpected database error")

    return PaperStatusResponse(
        paper_id=paper_id,
        status="pending",
    )


@router.get("/papers/{paper_id}", response_model=PaperStatusResponse)
async def get_paper_status(
    paper_id: str,
    _auth: None = Depends(_verify_api_key),
) -> PaperStatusResponse:
    """Get the current status and summary for a paper."""
    async with get_async_session() as session:
        paper = (await session.execute(select(Paper).where(Paper.id == paper_id))).scalars().first()

        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")

        report = (
            (
                await session.execute(
                    select(Report)
                    .where(Report.paper_id == paper_id)
                    .order_by(Report.created_at.desc())
                )
            )
            .scalars()
            .first()
        )

        return PaperStatusResponse(
            paper_id=str(paper.id),
            status=str(paper.status),
            risk_level=str(report.overall_risk) if report else None,
            overall_confidence=float(report.overall_confidence) if report else None,
            num_findings=int(report.num_findings)
            if report and report.num_findings is not None
            else None,
        )


@router.get("/papers/{paper_id}/report", response_model=ReportResponse)
async def get_paper_report(
    paper_id: str,
    _auth: None = Depends(_verify_api_key),
) -> ReportResponse:
    """Get the full analysis report for a paper."""
    async with get_async_session() as session:
        paper = (await session.execute(select(Paper).where(Paper.id == paper_id))).scalars().first()

        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")

        report = (
            (
                await session.execute(
                    select(Report)
                    .where(Report.paper_id == paper_id)
                    .order_by(Report.created_at.desc())
                )
            )
            .scalars()
            .first()
        )

        if not report:
            raise HTTPException(status_code=404, detail="Report not yet available for this paper")

        findings = (
            (await session.execute(select(Finding).where(Finding.paper_id == paper_id)))
            .scalars()
            .all()
        )

        finding_responses = [
            FindingResponse(
                id=str(f.id),
                analysis_type=str(f.analysis_type),
                severity=str(f.severity),
                confidence=float(f.confidence),
                title=str(f.title),
                description=str(f.description) if f.description else None,
            )
            for f in findings
        ]

        return ReportResponse(
            paper_id=paper_id,
            overall_risk=str(report.overall_risk),
            overall_confidence=float(report.overall_confidence),
            summary=str(report.summary) if report.summary else None,
            findings=finding_responses,
            converging_evidence=bool(report.converging_evidence),
        )


@router.post("/batch", status_code=202, response_model=BatchStatusResponse)
async def submit_batch(
    request: Request,
    body: BatchSubmitRequest,
    background_tasks: BackgroundTasks,
    _auth: None = Depends(_verify_api_key),
) -> BatchStatusResponse:
    """Submit a batch of DOIs for analysis."""
    if not body.dois:
        raise HTTPException(status_code=422, detail="Provide at least one DOI")

    # Validate all DOIs upfront
    validated_dois = []
    for doi in body.dois:
        try:
            validated_dois.append(validate_doi(doi))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"Invalid DOI {doi!r}: {e}")

    papers = []
    async with get_async_session() as session:
        for doi in validated_dois:
            existing = (
                (await session.execute(select(Paper).where(Paper.doi == doi))).scalars().first()
            )

            if existing:
                papers.append(
                    PaperStatusResponse(
                        paper_id=str(existing.id),
                        status=str(existing.status),
                    )
                )
                continue

            paper_id = str(uuid.uuid4())
            paper = Paper(
                id=paper_id,
                doi=doi,
                title=doi,
                source="api_batch",
                status="pending",
            )
            session.add(paper)
            papers.append(
                PaperStatusResponse(
                    paper_id=paper_id,
                    status="pending",
                )
            )
            background_tasks.add_task(_run_pipeline, request, paper_id)

    return BatchStatusResponse(papers=papers)


@router.get("/authors/{author_id}/risk", response_model=AuthorRiskResponse)
async def get_author_risk(
    author_id: str,
    _auth: None = Depends(_verify_api_key),
) -> AuthorRiskResponse:
    """Get the risk profile for an author."""
    async with get_async_session() as session:
        author = (
            (await session.execute(select(Author).where(Author.id == author_id))).scalars().first()
        )

        if not author:
            raise HTTPException(status_code=404, detail="Author not found")

        return AuthorRiskResponse(
            author_id=str(author.id),
            name=str(author.name),
            risk_score=float(author.risk_score) if author.risk_score is not None else None,
            total_papers=int(author.total_papers or 0),
            flagged_papers=int(author.flagged_papers or 0),
            retraction_count=int(author.retraction_count or 0),
        )
