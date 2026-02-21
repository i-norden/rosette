"""Initial schema: create all rosette tables.

Revision ID: 001
Revises: None
Create Date: 2025-01-01 00:00:00.000000

"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # papers
    # ------------------------------------------------------------------
    op.create_table(
        "papers",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("doi", sa.Text(), unique=True, nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column("authors_json", sa.Text(), nullable=True),
        sa.Column("journal", sa.Text(), nullable=True),
        sa.Column("journal_issn", sa.Text(), nullable=True),
        sa.Column("publication_year", sa.Integer(), nullable=True),
        sa.Column("citation_count", sa.Integer(), nullable=True),
        sa.Column("influential_citation_count", sa.Integer(), nullable=True),
        sa.Column("priority_score", sa.Float(), nullable=True),
        sa.Column("pdf_path", sa.Text(), nullable=True),
        sa.Column("pdf_sha256", sa.Text(), nullable=True),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), server_default="pending"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retraction_status", sa.Text(), nullable=True),
        sa.Column("retraction_date", sa.DateTime(), nullable=True),
        sa.Column("retraction_reason", sa.Text(), nullable=True),
        sa.Column("pubpeer_comments", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("idx_papers_status", "papers", ["status"])
    op.create_index("idx_papers_priority", "papers", [sa.text("priority_score DESC")])

    # ------------------------------------------------------------------
    # figures
    # ------------------------------------------------------------------
    op.create_table(
        "figures",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("paper_id", sa.Text(), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("figure_label", sa.Text(), nullable=True),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("image_path", sa.Text(), nullable=True),
        sa.Column("image_sha256", sa.Text(), nullable=True),
        sa.Column("image_type", sa.Text(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("phash", sa.Text(), nullable=True),
        sa.Column("ahash", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("idx_figures_paper", "figures", ["paper_id"])
    op.create_index("idx_figures_sha256", "figures", ["image_sha256"])
    op.create_index("idx_figures_phash", "figures", ["phash"])

    # ------------------------------------------------------------------
    # findings
    # ------------------------------------------------------------------
    op.create_table(
        "findings",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("paper_id", sa.Text(), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("figure_id", sa.Text(), sa.ForeignKey("figures.id"), nullable=True),
        sa.Column("analysis_type", sa.Text(), nullable=False),
        sa.Column("severity", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("evidence_json", sa.Text(), nullable=True),
        sa.Column("model_used", sa.Text(), nullable=True),
        sa.Column("raw_response", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("idx_findings_paper", "findings", ["paper_id"])
    op.create_index("idx_findings_severity", "findings", ["severity"])

    # ------------------------------------------------------------------
    # reports
    # ------------------------------------------------------------------
    op.create_table(
        "reports",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("paper_id", sa.Text(), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("overall_risk", sa.Text(), nullable=False),
        sa.Column("overall_confidence", sa.Float(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("report_markdown", sa.Text(), nullable=True),
        sa.Column("report_html", sa.Text(), nullable=True),
        sa.Column("num_findings", sa.Integer(), server_default="0"),
        sa.Column("num_critical", sa.Integer(), server_default="0"),
        sa.Column("converging_evidence", sa.Boolean(), server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )

    # ------------------------------------------------------------------
    # processing_log
    # ------------------------------------------------------------------
    op.create_table(
        "processing_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("paper_id", sa.Text(), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("details", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )

    # ------------------------------------------------------------------
    # authors
    # ------------------------------------------------------------------
    op.create_table(
        "authors",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("orcid", sa.Text(), unique=True, nullable=True),
        sa.Column("institution", sa.Text(), nullable=True),
        sa.Column("h_index", sa.Integer(), nullable=True),
        sa.Column("total_papers", sa.Integer(), server_default="0"),
        sa.Column("flagged_papers", sa.Integer(), server_default="0"),
        sa.Column("retraction_count", sa.Integer(), server_default="0"),
        sa.Column("risk_score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )

    # ------------------------------------------------------------------
    # author_paper_links
    # ------------------------------------------------------------------
    op.create_table(
        "author_paper_links",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("author_id", sa.Text(), sa.ForeignKey("authors.id"), nullable=False),
        sa.Column("paper_id", sa.Text(), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("position", sa.Integer(), nullable=True),
    )
    op.create_index("idx_author_paper_author", "author_paper_links", ["author_id"])
    op.create_index("idx_author_paper_paper", "author_paper_links", ["paper_id"])


def downgrade() -> None:
    op.drop_table("author_paper_links")
    op.drop_table("authors")
    op.drop_table("processing_log")
    op.drop_table("reports")
    op.drop_table("findings")
    op.drop_table("figures")
    op.drop_table("papers")
