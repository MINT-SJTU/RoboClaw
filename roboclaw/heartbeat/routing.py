"""Heartbeat delivery target selection."""

from __future__ import annotations

from typing import Any, Iterable


INTERNAL_CHANNELS = {"cli", "system"}


def pick_heartbeat_target(
    *,
    enabled_channels: Iterable[str],
    sessions: Iterable[dict[str, Any]],
    target_channel: str = "",
    target_chat_id: str = "",
) -> tuple[str, str]:
    """Pick a routable heartbeat target.

    Explicit config wins when both channel and chat id are set and the channel
    is enabled. Otherwise use the most recently updated external session.
    """
    enabled = set(enabled_channels)
    configured_channel = target_channel.strip()
    configured_chat_id = target_chat_id.strip()
    if configured_channel and configured_chat_id and configured_channel in enabled:
        return configured_channel, configured_chat_id

    for item in sessions:
        key = item.get("key") or ""
        if ":" not in key:
            continue
        channel, chat_id = key.split(":", 1)
        if channel in INTERNAL_CHANNELS:
            continue
        if channel in enabled and chat_id:
            return channel, chat_id
    return "cli", "direct"
