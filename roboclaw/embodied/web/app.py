"""FastAPI application for web-based data collection."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from roboclaw.embodied.web.routes import router
from roboclaw.embodied.web.session import RobotSession

_STATIC_DIR = Path(__file__).parent / "static"

# Singleton session shared across the app
_session = RobotSession()


def get_session() -> RobotSession:
    """Return the global robot session."""
    return _session


@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("Web data-collection server starting")
    yield
    logger.info("Web data-collection server shutting down")
    _session.disconnect()


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(title="RoboClaw Data Collection", lifespan=_lifespan)

    app.include_router(router)

    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/")
    async def index():
        index_path = _STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"message": "RoboClaw Data Collection API", "docs": "/docs"}

    return app
