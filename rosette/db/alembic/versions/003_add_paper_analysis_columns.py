"""Add paper analysis columns missing from initial schema.

Revision ID: 003
Revises: 002
Create Date: 2026-02-21 00:00:00.000000

"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("papers", sa.Column("full_text", sa.Text(), nullable=True))
    op.add_column("papers", sa.Column("risk_level", sa.Text(), nullable=True))
    op.add_column("papers", sa.Column("overall_confidence", sa.Float(), nullable=True))
    op.add_column("papers", sa.Column("converging_evidence", sa.Boolean(), nullable=True))


def downgrade() -> None:
    op.drop_column("papers", "converging_evidence")
    op.drop_column("papers", "overall_confidence")
    op.drop_column("papers", "risk_level")
    op.drop_column("papers", "full_text")
