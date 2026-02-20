"""SQLAlchemy ORM models for the rosette database."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[str] = mapped_column(primary_key=True, default=_uuid)
    doi: Mapped[str | None] = mapped_column(unique=True, default=None)
    title: Mapped[str] = mapped_column()
    abstract: Mapped[str | None] = mapped_column(default=None)
    authors_json: Mapped[str | None] = mapped_column(default=None)  # JSON array
    journal: Mapped[str | None] = mapped_column(default=None)
    journal_issn: Mapped[str | None] = mapped_column(default=None)
    publication_year: Mapped[int | None] = mapped_column(default=None)
    citation_count: Mapped[int | None] = mapped_column(default=None)
    influential_citation_count: Mapped[int | None] = mapped_column(default=None)
    priority_score: Mapped[float | None] = mapped_column(default=None)
    pdf_path: Mapped[str | None] = mapped_column(default=None)
    pdf_sha256: Mapped[str | None] = mapped_column(default=None)
    full_text: Mapped[str | None] = mapped_column(default=None)
    source: Mapped[str | None] = mapped_column(default=None)  # 'openalex', 'pubmed', 'manual', etc.
    status: Mapped[str] = mapped_column(default="pending")
    risk_level: Mapped[str | None] = mapped_column(default=None)  # critical/high/medium/low/clean
    overall_confidence: Mapped[float | None] = mapped_column(default=None)
    converging_evidence: Mapped[bool | None] = mapped_column(default=None)
    error_message: Mapped[str | None] = mapped_column(default=None)
    retraction_status: Mapped[str | None] = mapped_column(default=None)
    retraction_date: Mapped[datetime | None] = mapped_column(default=None)
    retraction_reason: Mapped[str | None] = mapped_column(default=None)
    pubpeer_comments: Mapped[int | None] = mapped_column(default=None)
    created_at: Mapped[datetime | None] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime | None] = mapped_column(default=_utcnow, onupdate=_utcnow)

    figures: Mapped[list[Figure]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    findings: Mapped[list[Finding]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    reports: Mapped[list[Report]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    processing_logs: Mapped[list[ProcessingLog]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_papers_status", "status"),
        Index("idx_papers_priority", priority_score.desc()),
    )


class Figure(Base):
    __tablename__ = "figures"

    id: Mapped[str] = mapped_column(primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"))
    page_number: Mapped[int | None] = mapped_column(default=None)
    figure_label: Mapped[str | None] = mapped_column(default=None)
    caption: Mapped[str | None] = mapped_column(default=None)
    image_path: Mapped[str | None] = mapped_column(default=None)
    image_sha256: Mapped[str | None] = mapped_column(default=None)
    image_type: Mapped[str | None] = mapped_column(
        default=None
    )  # 'western_blot', 'microscopy', etc.
    width: Mapped[int | None] = mapped_column(default=None)
    height: Mapped[int | None] = mapped_column(default=None)
    phash: Mapped[str | None] = mapped_column(default=None)
    ahash: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime | None] = mapped_column(default=_utcnow)

    paper: Mapped[Paper] = relationship(back_populates="figures")
    findings: Mapped[list[Finding]] = relationship(
        back_populates="figure", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_figures_paper", "paper_id"),
        Index("idx_figures_sha256", "image_sha256"),
        Index("idx_figures_phash", "phash"),
    )


class Finding(Base):
    __tablename__ = "findings"

    id: Mapped[str] = mapped_column(primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"))
    figure_id: Mapped[str | None] = mapped_column(ForeignKey("figures.id"), default=None)
    analysis_type: Mapped[str] = mapped_column()
    severity: Mapped[str] = mapped_column()  # 'critical', 'high', 'medium', 'low', 'info'
    confidence: Mapped[float] = mapped_column()
    title: Mapped[str] = mapped_column()
    description: Mapped[str | None] = mapped_column(default=None)
    evidence_json: Mapped[str | None] = mapped_column(default=None)  # JSON
    model_used: Mapped[str | None] = mapped_column(default=None)
    raw_response: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime | None] = mapped_column(default=_utcnow)

    paper: Mapped[Paper] = relationship(back_populates="findings")
    figure: Mapped[Figure | None] = relationship(back_populates="findings")

    __table_args__ = (
        Index("idx_findings_paper", "paper_id"),
        Index("idx_findings_severity", "severity"),
    )


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"))
    overall_risk: Mapped[str] = mapped_column()  # 'critical', 'high', 'medium', 'low', 'clean'
    overall_confidence: Mapped[float] = mapped_column()
    summary: Mapped[str | None] = mapped_column(default=None)
    report_markdown: Mapped[str | None] = mapped_column(default=None)
    report_html: Mapped[str | None] = mapped_column(default=None)
    num_findings: Mapped[int] = mapped_column(default=0)
    num_critical: Mapped[int] = mapped_column(default=0)
    converging_evidence: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime | None] = mapped_column(default=_utcnow)

    paper: Mapped[Paper] = relationship(back_populates="reports")


class ProcessingLog(Base):
    __tablename__ = "processing_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"))
    stage: Mapped[str] = mapped_column()
    status: Mapped[str] = mapped_column()  # 'started', 'completed', 'failed'
    details: Mapped[str | None] = mapped_column(default=None)
    started_at: Mapped[datetime | None] = mapped_column(default=None)
    completed_at: Mapped[datetime | None] = mapped_column(default=None)
    created_at: Mapped[datetime | None] = mapped_column(default=_utcnow)

    paper: Mapped[Paper] = relationship(back_populates="processing_logs")


class Author(Base):
    __tablename__ = "authors"

    id: Mapped[str] = mapped_column(primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column()
    orcid: Mapped[str | None] = mapped_column(unique=True, default=None)
    institution: Mapped[str | None] = mapped_column(default=None)
    h_index: Mapped[int | None] = mapped_column(default=None)
    total_papers: Mapped[int] = mapped_column(default=0)
    flagged_papers: Mapped[int] = mapped_column(default=0)
    retraction_count: Mapped[int] = mapped_column(default=0)
    risk_score: Mapped[float | None] = mapped_column(default=None)
    created_at: Mapped[datetime | None] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime | None] = mapped_column(default=_utcnow, onupdate=_utcnow)

    papers: Mapped[list[AuthorPaperLink]] = relationship(
        back_populates="author", cascade="all, delete-orphan"
    )


class AuthorPaperLink(Base):
    __tablename__ = "author_paper_links"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    author_id: Mapped[str] = mapped_column(ForeignKey("authors.id"))
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"))
    position: Mapped[int | None] = mapped_column(default=None)  # 1=first, -1=last

    author: Mapped[Author] = relationship(back_populates="papers")
    paper: Mapped[Paper] = relationship()

    __table_args__ = (
        Index("idx_author_paper_author", "author_id"),
        Index("idx_author_paper_paper", "paper_id"),
    )


class Campaign(Base):
    __tablename__ = "campaigns"

    id: Mapped[str] = mapped_column(primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column()
    mode: Mapped[str] = mapped_column()  # 'network_expansion', 'domain_scan', 'paper_mill'
    status: Mapped[str] = mapped_column(
        default="created"
    )  # created/seeding/auto_analyzing/expanding/llm_analyzing/paused/completed
    config_json: Mapped[str | None] = mapped_column(default=None)
    seed_dois: Mapped[str | None] = mapped_column(default=None)  # JSON array
    max_depth: Mapped[int] = mapped_column(default=2)
    max_papers: Mapped[int] = mapped_column(default=1000)
    llm_budget: Mapped[int] = mapped_column(default=100)
    papers_discovered: Mapped[int] = mapped_column(default=0)
    papers_triaged: Mapped[int] = mapped_column(default=0)
    papers_flagged: Mapped[int] = mapped_column(default=0)
    papers_llm_analyzed: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime | None] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime | None] = mapped_column(default=_utcnow, onupdate=_utcnow)

    campaign_papers: Mapped[list[CampaignPaper]] = relationship(
        back_populates="campaign", cascade="all, delete-orphan"
    )
    hash_matches: Mapped[list[ImageHashMatch]] = relationship(
        back_populates="campaign", cascade="all, delete-orphan"
    )


class CampaignPaper(Base):
    __tablename__ = "campaign_papers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    campaign_id: Mapped[str] = mapped_column(ForeignKey("campaigns.id"))
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"))
    source: Mapped[str] = (
        mapped_column()
    )  # 'seed', 'network_expansion', 'domain_scan', 'hash_match'
    source_paper_id: Mapped[str | None] = mapped_column(default=None)
    source_author_id: Mapped[str | None] = mapped_column(default=None)
    depth: Mapped[int] = mapped_column(default=0)
    triage_status: Mapped[str] = mapped_column(
        default="pending"
    )  # pending/auto_analyzing/auto_done/llm_queued/llm_analyzing/complete/dismissed
    auto_findings_count: Mapped[int] = mapped_column(default=0)
    auto_risk_score: Mapped[float | None] = mapped_column(default=None)
    llm_promoted: Mapped[bool] = mapped_column(default=False)
    final_risk: Mapped[str | None] = mapped_column(default=None)  # critical/high/medium/low/clean

    campaign: Mapped[Campaign] = relationship(back_populates="campaign_papers")
    paper: Mapped[Paper] = relationship()

    __table_args__ = (
        Index("idx_campaign_papers_campaign", "campaign_id"),
        Index("idx_campaign_papers_paper", "paper_id"),
        Index("idx_campaign_papers_triage", "triage_status"),
    )


class ImageHashMatch(Base):
    __tablename__ = "image_hash_matches"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    campaign_id: Mapped[str | None] = mapped_column(ForeignKey("campaigns.id"), default=None)
    figure_id_a: Mapped[str] = mapped_column(ForeignKey("figures.id"))
    figure_id_b: Mapped[str] = mapped_column(ForeignKey("figures.id"))
    paper_id_a: Mapped[str] = mapped_column(ForeignKey("papers.id"))
    paper_id_b: Mapped[str] = mapped_column(ForeignKey("papers.id"))
    hash_type: Mapped[str] = mapped_column()  # 'phash', 'ahash', 'sha256'
    hash_distance: Mapped[int] = mapped_column()
    verified: Mapped[bool | None] = mapped_column(default=None)  # None = unreviewed

    campaign: Mapped[Campaign | None] = relationship(back_populates="hash_matches")
    figure_a: Mapped[Figure] = relationship(foreign_keys=[figure_id_a])
    figure_b: Mapped[Figure] = relationship(foreign_keys=[figure_id_b])
    paper_a: Mapped[Paper] = relationship(foreign_keys=[paper_id_a])
    paper_b: Mapped[Paper] = relationship(foreign_keys=[paper_id_b])

    __table_args__ = (
        Index("idx_hash_matches_campaign", "campaign_id"),
        Index("idx_hash_matches_paper_a", "paper_id_a"),
        Index("idx_hash_matches_paper_b", "paper_id_b"),
    )
