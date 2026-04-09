from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from roboclaw.embodied.embodiment.interface.base import Interface


class InterfaceGuard:
    """Mutex guard for a hardware interface.

    Prevents concurrent access to the same physical device.
    Guard instances are managed by Manifest, keyed by interface.stable_id.
    """

    def __init__(self, interface: Interface) -> None:
        self._interface = interface
        self._lock = asyncio.Lock()
        self._owner: str = ""

    @property
    def interface(self) -> Interface:
        return self._interface

    @property
    def locked(self) -> bool:
        return self._lock.locked()

    @property
    def owner(self) -> str:
        return self._owner

    @asynccontextmanager
    async def acquire(self, owner: str) -> AsyncIterator[Interface]:
        """Acquire exclusive access. Yields the interface on success."""
        async with self._lock:
            self._owner = owner
            try:
                yield self._interface
            finally:
                self._owner = ""
