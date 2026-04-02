"""Web transport for the RoboClaw chat UI."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from contextlib import suppress
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from loguru import logger

from roboclaw.bus.events import OutboundMessage
from roboclaw.bus.queue import MessageBus
from roboclaw.channels.base import BaseChannel
from roboclaw.utils.helpers import timestamp


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
            if not content:
                continue
            await self._handle_message(
                sender_id=str(payload.get("sender_id") or user_id),
                chat_id=chat_id,
                content=content,
                media=payload.get("media") or [],
                metadata=payload.get("metadata") or {},
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
        content = json.dumps(content, ensure_ascii=False)
    return {
        "id": message.get("id") or f"{chat_id}:{index}",
        "role": message.get("role", "assistant"),
        "content": content,
        "timestamp": message.get("timestamp"),
        "metadata": {},
    }
