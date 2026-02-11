"""Alembic environment configuration for snoopy.

Supports both synchronous and asynchronous SQLAlchemy engines.  The database
URL is read from the snoopy configuration (``SnoopyConfig.storage.database_url``)
so that a single source of truth is used across the application.  If the
config cannot be loaded (e.g. during bare ``alembic`` CLI usage), the URL
falls back to the value in ``alembic.ini``.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from snoopy.db.models import Base

# Alembic Config object (provides access to alembic.ini values).
config = context.config

# Set up Python logging from the config file.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# MetaData object for 'autogenerate' support.
target_metadata = Base.metadata


def _get_url() -> str:
    """Return the database URL, preferring the snoopy config."""
    try:
        from snoopy.config import load_config

        cfg = load_config()
        return cfg.storage.database_url
    except Exception:
        # Fall back to alembic.ini value.
        return config.get_main_option("sqlalchemy.url", "sqlite:///snoopy.db")


# ---------------------------------------------------------------------------
# Offline (SQL-script) migrations
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This emits SQL statements to stdout rather than executing them against a
    live database connection.
    """
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online (synchronous engine) migrations
# ---------------------------------------------------------------------------

def run_migrations_online() -> None:
    """Run migrations against a live synchronous database connection."""
    url = _get_url()

    # If the URL is async-only (contains +aiosqlite / +asyncpg) hand off to
    # the async path instead.
    if "+aiosqlite" in url or "+asyncpg" in url:
        asyncio.run(run_migrations_async())
        return

    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = url

    connectable = engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


# ---------------------------------------------------------------------------
# Online (async engine) migrations
# ---------------------------------------------------------------------------

async def run_migrations_async() -> None:
    """Run migrations against a live asynchronous database connection."""
    url = _get_url()

    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = url

    connectable = async_engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(_do_run_migrations)

    await connectable.dispose()


def _do_run_migrations(connection) -> None:  # noqa: ANN001
    """Helper executed inside ``connection.run_sync``."""
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Entry point – Alembic calls whichever path matches the current mode.
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
