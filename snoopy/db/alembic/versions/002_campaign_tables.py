"""Add campaign investigation tables.

Revision ID: 002
Revises: 001
Create Date: 2025-01-15 00:00:00.000000

"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # campaigns
    # ------------------------------------------------------------------
    op.create_table(
        "campaigns",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("mode", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), server_default="created"),
        sa.Column("config_json", sa.Text(), nullable=True),
        sa.Column("seed_dois", sa.Text(), nullable=True),
        sa.Column("max_depth", sa.Integer(), server_default="2"),
        sa.Column("max_papers", sa.Integer(), server_default="1000"),
        sa.Column("llm_budget", sa.Integer(), server_default="100"),
        sa.Column("papers_discovered", sa.Integer(), server_default="0"),
        sa.Column("papers_triaged", sa.Integer(), server_default="0"),
        sa.Column("papers_flagged", sa.Integer(), server_default="0"),
        sa.Column("papers_llm_analyzed", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )

    # ------------------------------------------------------------------
    # campaign_papers
    # ------------------------------------------------------------------
    op.create_table(
        "campaign_papers",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("campaign_id", sa.Text(), sa.ForeignKey("campaigns.id"), nullable=False),
        sa.Column("paper_id", sa.Text(), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("source_paper_id", sa.Text(), nullable=True),
        sa.Column("source_author_id", sa.Text(), nullable=True),
        sa.Column("depth", sa.Integer(), server_default="0"),
        sa.Column("triage_status", sa.Text(), server_default="pending"),
        sa.Column("auto_findings_count", sa.Integer(), server_default="0"),
        sa.Column("auto_risk_score", sa.Float(), nullable=True),
        sa.Column("llm_promoted", sa.Boolean(), server_default="0"),
        sa.Column("final_risk", sa.Text(), nullable=True),
    )
    op.create_index("idx_campaign_papers_campaign", "campaign_papers", ["campaign_id"])
    op.create_index("idx_campaign_papers_paper", "campaign_papers", ["paper_id"])
    op.create_index("idx_campaign_papers_triage", "campaign_papers", ["triage_status"])

    # ------------------------------------------------------------------
    # image_hash_matches
    # ------------------------------------------------------------------
    op.create_table(
        "image_hash_matches",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("campaign_id", sa.Text(), sa.ForeignKey("campaigns.id"), nullable=True),
        sa.Column("figure_id_a", sa.Text(), sa.ForeignKey("figures.id"), nullable=False),
        sa.Column("figure_id_b", sa.Text(), sa.ForeignKey("figures.id"), nullable=False),
        sa.Column("paper_id_a", sa.Text(), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("paper_id_b", sa.Text(), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("hash_type", sa.Text(), nullable=False),
        sa.Column("hash_distance", sa.Integer(), nullable=False),
        sa.Column("verified", sa.Boolean(), nullable=True),
    )
    op.create_index("idx_hash_matches_campaign", "image_hash_matches", ["campaign_id"])
    op.create_index("idx_hash_matches_paper_a", "image_hash_matches", ["paper_id_a"])
    op.create_index("idx_hash_matches_paper_b", "image_hash_matches", ["paper_id_b"])


def downgrade() -> None:
    op.drop_table("image_hash_matches")
    op.drop_table("campaign_papers")
    op.drop_table("campaigns")
