"""Pydantic models for API request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PaperSubmitRequest(BaseModel):
    """Request body for submitting a paper for analysis."""

    doi: str | None = Field(default=None, description="DOI of the paper to analyze")
    pdf_upload: str | None = Field(
        default=None,
        max_length=70_000_000,
        description="Base64-encoded PDF content for file upload (~50 MB decoded limit)",
    )


class PaperStatusResponse(BaseModel):
    """Response for paper status queries."""

    paper_id: str
    status: str
    risk_level: str | None = None
    overall_confidence: float | None = None
    num_findings: int | None = None


class FindingResponse(BaseModel):
    """Individual finding in an analysis report."""

    id: str
    analysis_type: str
    severity: str
    confidence: float
    title: str
    description: str | None = None


class ReportResponse(BaseModel):
    """Full analysis report for a paper."""

    paper_id: str
    overall_risk: str
    overall_confidence: float
    summary: str | None = None
    findings: list[FindingResponse] = Field(default_factory=list)
    converging_evidence: bool = False


class BatchSubmitRequest(BaseModel):
    """Request body for batch DOI submission."""

    dois: list[str] = Field(description="List of DOIs to analyze", max_length=100)


class BatchStatusResponse(BaseModel):
    """Response for batch status queries."""

    papers: list[PaperStatusResponse] = Field(default_factory=list)


class AuthorRiskResponse(BaseModel):
    """Author risk profile response."""

    author_id: str
    name: str
    risk_score: float | None = None
    total_papers: int = 0
    flagged_papers: int = 0
    retraction_count: int = 0
