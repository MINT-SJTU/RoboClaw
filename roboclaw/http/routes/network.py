"""Network info route."""

from __future__ import annotations

import socket
from typing import Any, Callable

from fastapi import FastAPI
from loguru import logger


def _get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        logger.warning("Failed to detect LAN IP, falling back to 127.0.0.1")
        return "127.0.0.1"


def register_network_routes(
    app: FastAPI,
    get_config: Callable[[], tuple[str, int]],
) -> None:

    @app.get("/api/system/network")
    async def network_info() -> dict[str, Any]:
        host, port = get_config()
        return {"host": host, "port": port, "lan_ip": _get_lan_ip()}
