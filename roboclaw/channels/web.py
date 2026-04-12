"""Web transport for the RoboClaw chat UI."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import mimetypes
import re
from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

from roboclaw.bus.events import OutboundMessage
from roboclaw.bus.queue import MessageBus
from roboclaw.channels.base import BaseChannel
from roboclaw.utils.helpers import detect_image_mime, ensure_dir, safe_filename, timestamp

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


class WebChannel(BaseChannel):
    """WebSocket channel for the RoboClaw web UI.

    Pure transport — does not own a FastAPI app.  Call
    ``register_routes(app)`` to wire the endpoints onto an external app.
    """

    name = "web"
    display_name = "Web UI"

    def __init__(self, config: Any, bus: MessageBus, session_manager: Any | None = None):
        super().__init__(config, bus)
        self.sessions = session_manager
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._stop_event = asyncio.Event()

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """Default local-first config for the web UI."""
        return {
            "enabled": False,
            "allow_from": ["*"],
            "host": "127.0.0.1",
            "port": 8765,
            "cors_origins": ["http://localhost:5173"],
        }

    # ------------------------------------------------------------------
    # Route registration
    # ------------------------------------------------------------------

    def register_routes(self, app: FastAPI) -> None:
        """Register chat websocket and session REST routes on *app*."""

        @app.get("/api/health")
        async def health_check() -> dict[str, str]:
            return {"status": "ok", "channel": self.name}

        @app.get("/api/chat/sessions")
        async def list_sessions() -> list[dict[str, Any]]:
            if self.sessions is None:
                return []
            return [
                _session_summary(item)
                for item in self.sessions.list_sessions()
                if item.get("key", "").startswith(f"{self.name}:")
            ]

        @app.get("/api/chat/sessions/{chat_id}")
        async def get_session(chat_id: str) -> dict[str, Any]:
            if self.sessions is None:
                raise HTTPException(status_code=404, detail="Session storage is unavailable.")
            return {"chat_id": chat_id, "messages": self._session_history(chat_id)}

        @app.post("/api/chat/uploads/image")
        async def upload_chat_image(payload: ChatImageUploadRequest) -> dict[str, Any]:
            if self.sessions is None:
                raise HTTPException(status_code=503, detail="Session storage is unavailable.")
            return self._save_chat_image(
                chat_id=payload.chat_id,
                data_url=payload.data_url,
                original_name=payload.filename,
            )

        @app.get("/api/chat/uploads/{chat_id}/{file_name}")
        async def get_chat_upload(chat_id: str, file_name: str) -> FileResponse:
            if self.sessions is None:
                raise HTTPException(status_code=404, detail="Session storage is unavailable.")
            file_path = self._resolve_chat_upload_path(chat_id, file_name)
            if not file_path.is_file():
                raise HTTPException(status_code=404, detail="Uploaded image not found.")
            media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            return FileResponse(str(file_path), media_type=media_type, filename=file_path.name)

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            chat_id = websocket.query_params.get("chat_id") or uuid4().hex[:12]
            await self._handle_websocket(websocket, chat_id)

    # ------------------------------------------------------------------
    # WebSocket handling
    # ------------------------------------------------------------------

    async def _handle_websocket(self, websocket: WebSocket, chat_id: str) -> None:
        """Accept a socket and proxy messages into the shared bus."""
        await websocket.accept()
        self._connections[chat_id].add(websocket)
        user_id = f"web:{chat_id}"
        logger.info("Web UI client connected: {}", user_id)

        await self._send_json(
            websocket,
            {
                "type": "session.init",
                "chat_id": chat_id,
                "history": self._session_history(chat_id),
            },
        )

        try:
            await self._read_loop(websocket, chat_id, user_id)
        except WebSocketDisconnect:
            logger.info("Web UI client disconnected: {}", user_id)
        except json.JSONDecodeError:
            await self._send_json(
                websocket,
                {"type": "error", "code": "parse_error", "message": "Expected JSON websocket payload."},
            )
        except Exception:
            logger.exception("WebSocket error for {}", user_id)
        finally:
            self._connections[chat_id].discard(websocket)
            if not self._connections[chat_id]:
                self._connections.pop(chat_id, None)

    async def _read_loop(self, websocket: WebSocket, chat_id: str, user_id: str) -> None:
        while True:
            payload = json.loads(await websocket.receive_text())
            msg_type = payload.get("type", "chat.send")
            if msg_type != "chat.send":
                await self._send_json(websocket, {
                    "type": "error",
                    "code": "unknown_type",
                    "message": f"Unknown message type: {msg_type}",
                })
                continue
            content = str(payload.get("content", "")).strip()
            media = [str(item) for item in (payload.get("media") or []) if str(item).strip()]
            metadata = payload.get("metadata") or {}
            if not content and not media:
                continue
            if not content:
                content = "[image]"
            await self._handle_message(
                sender_id=str(payload.get("sender_id") or user_id),
                chat_id=chat_id,
                content=content,
                media=media,
                metadata=metadata,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _send_json(self, websocket: WebSocket, payload: dict[str, Any]) -> None:
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))

    def _session_key(self, chat_id: str) -> str:
        return f"{self.name}:{chat_id}"

    def _session_history(self, chat_id: str) -> list[dict[str, Any]]:
        if self.sessions is None:
            return []
        session = self.sessions.get_or_create(self._session_key(chat_id))
        return [
            _history_entry(chat_id, index, message)
            for index, message in enumerate(session.messages)
        ]

    def _chat_upload_root(self) -> Path:
        if self.sessions is None:
            raise RuntimeError("Session storage is unavailable.")
        return ensure_dir(self.sessions.workspace / "chat_uploads" / self.name)

    def _chat_upload_dir(self, chat_id: str) -> Path:
        return ensure_dir(self._chat_upload_root() / safe_filename(chat_id))

    def _resolve_chat_upload_path(self, chat_id: str, file_name: str) -> Path:
        chat_dir = self._chat_upload_dir(chat_id).resolve()
        file_path = (chat_dir / safe_filename(file_name)).resolve()
        if not str(file_path).startswith(str(chat_dir)):
            raise HTTPException(status_code=403, detail="Path traversal not allowed.")
        return file_path

    def _save_chat_image(
        self,
        *,
        chat_id: str,
        data_url: str,
        original_name: str | None,
    ) -> dict[str, Any]:
        if len(data_url) > _CHAT_IMAGE_MAX_BYTES * 2:  # base64 ~1.37x overhead + header
            raise HTTPException(status_code=413, detail="Image exceeds 8 MB limit.")

        match = _DATA_URL_RE.match(data_url.strip())
        if not match:
            raise HTTPException(status_code=400, detail="Expected a base64 image data URL.")

        try:
            raw = base64.b64decode(match.group("data"), validate=True)
        except (ValueError, binascii.Error):  # type: ignore[name-defined]
            raise HTTPException(status_code=400, detail="Invalid base64 image payload.") from None

        if not raw:
            raise HTTPException(status_code=400, detail="Image payload is empty.")
        if len(raw) > _CHAT_IMAGE_MAX_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds 8 MB limit.")

        detected_mime = detect_image_mime(raw)
        declared_mime = match.group("mime")
        mime = detected_mime or declared_mime
        if mime not in _MIME_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported image format.")

        original_path = Path(original_name or "image")
        stem = safe_filename(original_path.stem) or "image"
        extension = _MIME_EXTENSIONS[mime]
        stored_name = f"{uuid4().hex[:12]}-{stem}{extension}"
        target_path = self._chat_upload_dir(chat_id) / stored_name
        target_path.write_bytes(raw)

        return {
            "id": uuid4().hex[:12],
            "name": original_path.name or f"{stem}{extension}",
            "preview_url": f"/api/chat/uploads/{chat_id}/{stored_name}",
            "mime_type": mime,
            "size": len(raw),
        }

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------

    async def broadcast(self, event: dict[str, Any]) -> None:
        """Send an event to all connected WebSocket clients."""
        all_sockets = [
            ws
            for sockets in self._connections.values()
            for ws in sockets
        ]
        if not all_sockets:
            return

        async def _send_safe(ws: WebSocket) -> None:
            try:
                await self._send_json(ws, event)
            except (ConnectionError, RuntimeError, WebSocketDisconnect):
                pass

        await asyncio.gather(*[_send_safe(ws) for ws in all_sockets])

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send(self, msg: OutboundMessage) -> None:
        """Broadcast an outbound message to all sockets bound to the chat id."""
        sockets = list(self._connections.get(msg.chat_id, ()))
        if not sockets:
            return

        payload = {
            "type": "chat.message",
            "chat_id": msg.chat_id,
            "role": "assistant",
            "content": msg.content,
            "timestamp": timestamp(),
            "metadata": msg.metadata or {},
        }

        disconnected: list[WebSocket] = []
        for websocket in sockets:
            try:
                await self._send_json(websocket, payload)
            except (ConnectionError, RuntimeError, WebSocketDisconnect) as exc:
                logger.warning("Failed to send websocket message: {}", exc)
                disconnected.append(websocket)

        for websocket in disconnected:
            self._connections[msg.chat_id].discard(websocket)
        if msg.chat_id in self._connections and not self._connections[msg.chat_id]:
            self._connections.pop(msg.chat_id, None)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Keep the channel alive for ChannelManager compatibility."""
        self._running = True
        self._stop_event.clear()
        await self._stop_event.wait()

    async def stop(self) -> None:
        """Stop the channel and close open sockets."""
        self._running = False
        self._stop_event.set()

        for sockets in list(self._connections.values()):
            for websocket in list(sockets):
                with suppress(ConnectionError, RuntimeError, WebSocketDisconnect):
                    await websocket.close()
        self._connections.clear()


# ------------------------------------------------------------------
# Module-level helpers (keep dict construction out of nested scopes)
# ------------------------------------------------------------------


def _session_summary(item: dict[str, Any]) -> dict[str, Any]:
    key = item.get("key", "")
    return {
        "chat_id": key.split(":", 1)[1],
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
    }


def _history_entry(chat_id: str, index: int, message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = [
            str(block.get("text", "")).strip()
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        content = "\n".join(part for part in text_parts if part).strip() or "[image]"
    return {
        "id": message.get("id") or f"{chat_id}:{index}",
        "role": message.get("role", "assistant"),
        "content": content,
        "timestamp": message.get("timestamp"),
        "metadata": message.get("metadata") or {},
    }
