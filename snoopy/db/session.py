"""Database session management and initialization."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from snoopy.db.models import Base


_engine = None
_SessionFactory = None

_async_engine = None
_AsyncSessionFactory = None


def reset_db() -> None:
    """Reset all database state.  Intended for use in test teardown."""
    global _engine, _SessionFactory, _async_engine, _AsyncSessionFactory
    if _engine is not None:
        _engine.dispose()
    if _async_engine is not None:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_async_engine.dispose())
        except RuntimeError:
            asyncio.run(_async_engine.dispose())
    _engine = None
    _SessionFactory = None
    _async_engine = None
    _AsyncSessionFactory = None


def init_db(database_url: str = "sqlite:///snoopy.db") -> None:
    """Initialize the database engine and create all tables.

    WARNING: Must be called once at startup before any concurrent access.
    Not safe to call from multiple threads simultaneously.
    """
    global _engine, _SessionFactory

    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    _engine = create_engine(database_url, connect_args=connect_args)
    _SessionFactory = sessionmaker(bind=_engine)
    Base.metadata.create_all(_engine)


def init_async_db(database_url: str = "sqlite:///snoopy.db") -> None:
    """Initialize the async database engine for use in async contexts.

    Converts standard database URLs to their async equivalents
    (e.g. sqlite:// -> sqlite+aiosqlite://).

    WARNING: Must be called once at startup before any concurrent access.
    Not safe to call from multiple threads simultaneously.
    """
    global _async_engine, _AsyncSessionFactory

    async_url = database_url
    if async_url.startswith("sqlite://") and "+aiosqlite" not in async_url:
        async_url = async_url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    elif async_url.startswith("postgresql://") and "+asyncpg" not in async_url:
        async_url = async_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    _async_engine = create_async_engine(async_url)
    _AsyncSessionFactory = async_sessionmaker(bind=_async_engine, class_=AsyncSession)


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


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional async database session."""
    if _AsyncSessionFactory is None:
        raise RuntimeError("Async database not initialized. Call init_async_db() first.")
    session = _AsyncSessionFactory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
