"""Shared helpers for camera preview route handlers."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse


def serve_preview_image(preview_dir: Path, preview_key: str) -> FileResponse:
    """Return the preview JPEG for *preview_key* or raise HTTP 404."""
    preview_path = preview_dir / f"{preview_key}.jpg"
    if not preview_path.exists():
        raise HTTPException(404, f"Preview not found for key {preview_key}")
    return FileResponse(
        str(preview_path),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )
