"""EvoMind JWT authentication middleware and username resolution.

Auth modes (EVO_STUDIO_AUTH_MODE):
  dev            - trust x-evo-studio-user header or username param (default)
  trusted_header - trust x-evo-studio-user injected by upstream gateway
  evomind        - validate Bearer token against EvoMind cloud API
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import httpx

_log = logging.getLogger(__name__)

_EVO_API = "https://api.evomind-tech.com"
_CACHE_TTL = 300  # seconds — reuse verified identity for 5 minutes
_cache: dict[str, tuple[str, float]] = {}  # token -> (username, expires_at)
_cache_lock = threading.Lock()


def _auth_mode() -> str:
    return os.environ.get("EVO_STUDIO_AUTH_MODE", "dev").strip().lower()


def _cached_username(token: str) -> str | None:
    with _cache_lock:
        entry = _cache.get(token)
    if entry and entry[1] > time.monotonic():
        return entry[0]
    return None


def _store_cached_username(token: str, username: str) -> None:
    with _cache_lock:
        _cache[token] = (username, time.monotonic() + _CACHE_TTL)


def _evomind_username(token: str) -> str:
    """Validate token against EvoMind API and return username.

    Uses phone number as username since EvoMind uses phone-based auth.
    Raises ValueError if token is invalid or API is unreachable.
    """
    cached = _cached_username(token)
    if cached is not None:
        return cached

    try:
        response = httpx.get(
            f"{_EVO_API}/auth/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (401, 403):
            raise ValueError("invalid or expired EvoMind token") from exc
        raise ValueError(f"EvoMind auth API error: {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise ValueError(f"EvoMind auth API unreachable: {exc}") from exc

    # Use nickname if set, otherwise phone number
    username = str(data.get("nickname") or data.get("phone") or "").strip()
    if not username:
        raise ValueError("EvoMind token valid but user has no identifier")

    _store_cached_username(token, username)
    return username


def resolve_username(
    provided: str = "",
    header_username: str = "",
    bearer_token: str = "",
) -> str:
    """Resolve the effective username for a request.

    Resolution order by auth mode:
      dev:            provided > header_username (no token validation)
      trusted_header: header_username only (gateway-injected, trusted)
      evomind:        bearer_token validated against EvoMind API
    """
    mode = _auth_mode()

    if mode == "trusted_header":
        return header_username.strip()

    if mode == "evomind":
        token = bearer_token.strip()
        if not token:
            return ""
        try:
            return _evomind_username(token)
        except ValueError as exc:
            _log.warning("EvoMind token validation failed: %s", exc)
            return ""

    # dev mode — trust whatever is provided
    return (provided or header_username).strip()


def require_username(
    provided: str = "",
    header_username: str = "",
    bearer_token: str = "",
) -> str:
    """Like resolve_username but raises ValueError if result is empty."""
    username = resolve_username(provided, header_username, bearer_token)
    if not username:
        mode = _auth_mode()
        if mode == "evomind":
            raise ValueError("authentication required: provide a valid EvoMind Bearer token")
        raise ValueError("username is required")
    return username


def clear_token_cache_for_tests() -> None:
    with _cache_lock:
        _cache.clear()
