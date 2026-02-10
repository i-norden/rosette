"""Database session management and initialization."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from snoopy.db.models import Base


_engine = None
_SessionFactory = None


def init_db(database_url: str = "sqlite:///snoopy.db") -> None:
    """Initialize the database engine and create all tables."""
    global _engine, _SessionFactory

    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    _engine = create_engine(database_url, connect_args=connect_args)
    _SessionFactory = sessionmaker(bind=_engine)
    Base.metadata.create_all(_engine)


def get_engine():
    """Return the current database engine."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _engine


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional database session."""
    if _SessionFactory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
