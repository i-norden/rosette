"""SQLAlchemy ORM models for the snoopy database."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    id = Column(Text, primary_key=True, default=_uuid)
    doi = Column(Text, unique=True, nullable=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=True)
    authors_json = Column(Text, nullable=True)  # JSON array
    journal = Column(Text, nullable=True)
    journal_issn = Column(Text, nullable=True)
    publication_year = Column(Integer, nullable=True)
    citation_count = Column(Integer, nullable=True)
    influential_citation_count = Column(Integer, nullable=True)
    priority_score = Column(Float, nullable=True)
    pdf_path = Column(Text, nullable=True)
    pdf_sha256 = Column(Text, nullable=True)
    full_text = Column(Text, nullable=True)
    source = Column(Text, nullable=True)  # 'openalex', 'pubmed', 'manual', etc.
    status = Column(Text, default="pending")
    risk_level = Column(Text, nullable=True)  # Aggregated risk: critical/high/medium/low/clean
    overall_confidence = Column(Float, nullable=True)  # Aggregated confidence score
    converging_evidence = Column(Boolean, nullable=True)  # Whether multiple methods converge
    error_message = Column(Text, nullable=True)
    retraction_status = Column(Text, nullable=True)  # 'retracted', 'expression_of_concern', etc.
    retraction_date = Column(DateTime, nullable=True)
    retraction_reason = Column(Text, nullable=True)
    pubpeer_comments = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    figures = relationship("Figure", back_populates="paper", cascade="all, delete-orphan")
    findings = relationship("Finding", back_populates="paper", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="paper", cascade="all, delete-orphan")
    processing_logs = relationship(
        "ProcessingLog", back_populates="paper", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_papers_status", "status"),
        Index("idx_papers_priority", priority_score.desc()),
    )


class Figure(Base):
    __tablename__ = "figures"

    id = Column(Text, primary_key=True, default=_uuid)
    paper_id = Column(Text, ForeignKey("papers.id"), nullable=False)
    page_number = Column(Integer, nullable=True)
    figure_label = Column(Text, nullable=True)
    caption = Column(Text, nullable=True)
    image_path = Column(Text, nullable=True)
    image_sha256 = Column(Text, nullable=True)
    image_type = Column(Text, nullable=True)  # 'western_blot', 'microscopy', 'gel', etc.
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    phash = Column(Text, nullable=True)  # Perceptual hash for cross-reference
    ahash = Column(Text, nullable=True)  # Average hash for cross-reference
    created_at = Column(DateTime, default=_utcnow)

    paper = relationship("Paper", back_populates="figures")
    findings = relationship("Finding", back_populates="figure", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_figures_paper", "paper_id"),
        Index("idx_figures_sha256", "image_sha256"),
        Index("idx_figures_phash", "phash"),
    )


class Finding(Base):
    __tablename__ = "findings"

    id = Column(Text, primary_key=True, default=_uuid)
    paper_id = Column(Text, ForeignKey("papers.id"), nullable=False)
    figure_id = Column(Text, ForeignKey("figures.id"), nullable=True)
    analysis_type = Column(Text, nullable=False)
    severity = Column(Text, nullable=False)  # 'critical', 'high', 'medium', 'low', 'info'
    confidence = Column(Float, nullable=False)
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    evidence_json = Column(Text, nullable=True)  # JSON
    model_used = Column(Text, nullable=True)
    raw_response = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    paper = relationship("Paper", back_populates="findings")
    figure = relationship("Figure", back_populates="findings")

    __table_args__ = (
        Index("idx_findings_paper", "paper_id"),
        Index("idx_findings_severity", "severity"),
    )


class Report(Base):
    __tablename__ = "reports"

    id = Column(Text, primary_key=True, default=_uuid)
    paper_id = Column(Text, ForeignKey("papers.id"), nullable=False)
    overall_risk = Column(Text, nullable=False)  # 'critical', 'high', 'medium', 'low', 'clean'
    overall_confidence = Column(Float, nullable=False)
    summary = Column(Text, nullable=True)
    report_markdown = Column(Text, nullable=True)
    report_html = Column(Text, nullable=True)
    num_findings = Column(Integer, default=0)
    num_critical = Column(Integer, default=0)
    converging_evidence = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_utcnow)

    paper = relationship("Paper", back_populates="reports")


class ProcessingLog(Base):
    __tablename__ = "processing_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(Text, ForeignKey("papers.id"), nullable=False)
    stage = Column(Text, nullable=False)
    status = Column(Text, nullable=False)  # 'started', 'completed', 'failed'
    details = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    paper = relationship("Paper", back_populates="processing_logs")


class Author(Base):
    __tablename__ = "authors"

    id = Column(Text, primary_key=True, default=_uuid)
    name = Column(Text, nullable=False)
    orcid = Column(Text, nullable=True, unique=True)
    institution = Column(Text, nullable=True)
    h_index = Column(Integer, nullable=True)
    total_papers = Column(Integer, default=0)
    flagged_papers = Column(Integer, default=0)
    retraction_count = Column(Integer, default=0)
    risk_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    papers = relationship("AuthorPaperLink", back_populates="author", cascade="all, delete-orphan")


class AuthorPaperLink(Base):
    __tablename__ = "author_paper_links"

    id = Column(Integer, primary_key=True, autoincrement=True)
    author_id = Column(Text, ForeignKey("authors.id"), nullable=False)
    paper_id = Column(Text, ForeignKey("papers.id"), nullable=False)
    position = Column(Integer, nullable=True)  # Author position (1=first, -1=last)

    author = relationship("Author", back_populates="papers")
    paper = relationship("Paper")

    __table_args__ = (
        Index("idx_author_paper_author", "author_id"),
        Index("idx_author_paper_paper", "paper_id"),
    )


class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(Text, primary_key=True, default=_uuid)
    name = Column(Text, nullable=False)
    mode = Column(Text, nullable=False)  # 'network_expansion', 'domain_scan', 'paper_mill'
    status = Column(
        Text, default="created"
    )  # created/seeding/auto_analyzing/expanding/llm_analyzing/paused/completed
    config_json = Column(Text, nullable=True)  # JSON: mode-specific parameters
    seed_dois = Column(Text, nullable=True)  # JSON array of seed DOIs
    max_depth = Column(Integer, default=2)
    max_papers = Column(Integer, default=1000)
    llm_budget = Column(Integer, default=100)
    papers_discovered = Column(Integer, default=0)
    papers_triaged = Column(Integer, default=0)
    papers_flagged = Column(Integer, default=0)
    papers_llm_analyzed = Column(Integer, default=0)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    campaign_papers = relationship(
        "CampaignPaper", back_populates="campaign", cascade="all, delete-orphan"
    )
    hash_matches = relationship(
        "ImageHashMatch", back_populates="campaign", cascade="all, delete-orphan"
    )


class CampaignPaper(Base):
    __tablename__ = "campaign_papers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    campaign_id = Column(Text, ForeignKey("campaigns.id"), nullable=False)
    paper_id = Column(Text, ForeignKey("papers.id"), nullable=False)
    source = Column(
        Text, nullable=False
    )  # 'seed', 'network_expansion', 'domain_scan', 'hash_match'
    source_paper_id = Column(Text, nullable=True)  # Paper that triggered expansion
    source_author_id = Column(Text, nullable=True)  # Author that linked them
    depth = Column(Integer, default=0)  # Expansion depth from seed (0 for seeds)
    triage_status = Column(
        Text, default="pending"
    )  # pending/auto_analyzing/auto_done/llm_queued/llm_analyzing/complete/dismissed
    auto_findings_count = Column(Integer, default=0)
    auto_risk_score = Column(Float, nullable=True)  # 0-100
    llm_promoted = Column(Boolean, default=False)
    final_risk = Column(Text, nullable=True)  # critical/high/medium/low/clean

    campaign = relationship("Campaign", back_populates="campaign_papers")
    paper = relationship("Paper")

    __table_args__ = (
        Index("idx_campaign_papers_campaign", "campaign_id"),
        Index("idx_campaign_papers_paper", "paper_id"),
        Index("idx_campaign_papers_triage", "triage_status"),
    )


class ImageHashMatch(Base):
    __tablename__ = "image_hash_matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    campaign_id = Column(Text, ForeignKey("campaigns.id"), nullable=True)
    figure_id_a = Column(Text, ForeignKey("figures.id"), nullable=False)
    figure_id_b = Column(Text, ForeignKey("figures.id"), nullable=False)
    paper_id_a = Column(Text, ForeignKey("papers.id"), nullable=False)
    paper_id_b = Column(Text, ForeignKey("papers.id"), nullable=False)
    hash_type = Column(Text, nullable=False)  # 'phash', 'ahash', 'sha256'
    hash_distance = Column(Integer, nullable=False)
    verified = Column(Boolean, nullable=True)  # None = unreviewed

    campaign = relationship("Campaign", back_populates="hash_matches")
    figure_a = relationship("Figure", foreign_keys=[figure_id_a])
    figure_b = relationship("Figure", foreign_keys=[figure_id_b])
    paper_a = relationship("Paper", foreign_keys=[paper_id_a])
    paper_b = relationship("Paper", foreign_keys=[paper_id_b])

    __table_args__ = (
        Index("idx_hash_matches_campaign", "campaign_id"),
        Index("idx_hash_matches_paper_a", "paper_id_a"),
        Index("idx_hash_matches_paper_b", "paper_id_b"),
    )
