"""FastAPI application factory for the snoopy API server."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from snoopy.config import SnoopyConfig
from snoopy.db.session import init_async_db, init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialize resources on startup and clean up on shutdown."""
    config = app.state.config

    init_db(config.storage.database_url)
    init_async_db(config.storage.database_url)
    logger.info("Database initialized for API server")

    try:
        from snoopy.pipeline.orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(config)
        app.state.orchestrator = orchestrator
        logger.info("Pipeline orchestrator initialized")
    except Exception as e:
        logger.warning("Could not initialize orchestrator: %s", e)
        app.state.orchestrator = None

    yield


def create_app(config: SnoopyConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Optional SnoopyConfig instance. If not provided, default
            configuration is used.

    Returns:
        A configured FastAPI application.
    """
    if config is None:
        from snoopy.config import load_config

        config = load_config()

    app = FastAPI(
        title="Snoopy API",
        description="LLM-powered academic integrity analyzer",
        version="0.1.0",
        lifespan=_lifespan,
    )

    # Store config on app.state so routes can access via request.app.state
    app.state.config = config
    app.state.orchestrator = None

    # --- Rate limiting ---
    limiter = Limiter(key_func=get_remote_address, default_limits=[config.rate_limit])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # --- CORS middleware ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    )

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    # Include routes
    from snoopy.api.routes import router

    app.include_router(router)

    return app
