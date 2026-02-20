"""Schema creation and lightweight migration support."""

from __future__ import annotations

from sqlalchemy import inspect, text

from rosette.db.models import Base
from rosette.db.session import get_engine


def create_all_tables() -> None:
    """Create all tables that don't already exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)


def check_schema() -> dict[str, bool]:
    """Check which expected tables exist in the database."""
    engine = get_engine()
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    expected = set(Base.metadata.tables.keys())
    return {table: table in existing for table in expected}


def reset_database() -> None:
    """Drop and recreate all tables. USE WITH CAUTION."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def get_paper_counts() -> dict[str, int]:
    """Get paper counts by status for monitoring."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT status, COUNT(*) FROM papers GROUP BY status"))
        return dict(result.fetchall())
