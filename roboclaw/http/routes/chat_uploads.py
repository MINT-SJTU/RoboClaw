"""Chat image upload routes — serves the web frontend's attachment feature."""

from __future__ import annotations

import base64
import binascii
import mimetypes
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from roboclaw.config.paths import get_workspace_path
from roboclaw.utils.helpers import detect_image_mime, ensure_dir, safe_filename

_DATA_URL_RE = re.compile(r"^data:(?P<mime>image/[a-zA-Z0-9.+-]+);base64,(?P<data>.+)$")
_CHAT_IMAGE_MAX_BYTES = 8 * 1024 * 1024
_MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


class ChatImageUploadRequest(BaseModel):
    chat_id: str
    data_url: str
    filename: str | None = None


def _upload_dir(chat_id: str) -> Path:
    return ensure_dir(get_workspace_path() / "chat_uploads" / safe_filename(chat_id))


def _resolve_upload_path(chat_id: str, file_name: str) -> Path:
    chat_dir = _upload_dir(chat_id).resolve()
    file_path = (chat_dir / safe_filename(file_name)).resolve()
    if not str(file_path).startswith(str(chat_dir)):
        raise HTTPException(status_code=403, detail="Path traversal not allowed.")
    return file_path


def _save_image(*, chat_id: str, data_url: str, original_name: str | None) -> dict[str, Any]:
    if len(data_url) > _CHAT_IMAGE_MAX_BYTES * 2:
        raise HTTPException(status_code=413, detail="Image exceeds 8 MB limit.")

    match = _DATA_URL_RE.match(data_url.strip())
    if not match:
        raise HTTPException(status_code=400, detail="Expected a base64 image data URL.")

    try:
        raw = base64.b64decode(match.group("data"), validate=True)
    except (ValueError, binascii.Error):
        raise HTTPException(status_code=400, detail="Invalid base64 image payload.") from None

    if not raw:
        raise HTTPException(status_code=400, detail="Image payload is empty.")
    if len(raw) > _CHAT_IMAGE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 8 MB limit.")

    detected_mime = detect_image_mime(raw)
    if not detected_mime:
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    if detected_mime not in _MIME_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    original_path = Path(original_name or "image")
    stem = safe_filename(original_path.stem) or "image"
    extension = _MIME_EXTENSIONS[detected_mime]
    file_id = uuid4().hex[:12]
    stored_name = f"{file_id}-{stem}{extension}"
    target_path = _upload_dir(chat_id) / stored_name
    target_path.write_bytes(raw)

    return {
        "id": file_id,
        "name": original_path.name or f"{stem}{extension}",
        "preview_url": f"/api/chat/uploads/{chat_id}/{stored_name}",
        "mime_type": detected_mime,
        "size": len(raw),
    }


def register_chat_upload_routes(app: FastAPI) -> None:
    @app.post("/api/chat/uploads/image")
    async def upload_chat_image(payload: ChatImageUploadRequest) -> dict[str, Any]:
        return _save_image(
            chat_id=payload.chat_id,
            data_url=payload.data_url,
            original_name=payload.filename,
        )

    @app.get("/api/chat/uploads/{chat_id}/{file_name}")
    async def get_chat_upload(chat_id: str, file_name: str) -> FileResponse:
        file_path = _resolve_upload_path(chat_id, file_name)
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="Uploaded image not found.")
        media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        return FileResponse(str(file_path), media_type=media_type, filename=file_path.name)
