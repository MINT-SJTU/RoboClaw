"""Cooperative serial port lock registry.

All serial port access across the application should go through this module
to prevent concurrent access to the same physical device.
"""

import asyncio
import threading
from contextlib import asynccontextmanager
from typing import AsyncIterator


class PortLockRegistry:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._thread_lock = threading.Lock()  # protects _locks dict creation only

    def _get_lock(self, port: str) -> asyncio.Lock:
        with self._thread_lock:
            if port not in self._locks:
                self._locks[port] = asyncio.Lock()
            return self._locks[port]

    @asynccontextmanager
    async def acquire(self, port: str) -> AsyncIterator[None]:
        lock = self._get_lock(port)
        async with lock:
            yield

    @asynccontextmanager
    async def acquire_many(self, ports: list[str]) -> AsyncIterator[None]:
        sorted_ports = sorted(set(ports))
        locks = [self._get_lock(p) for p in sorted_ports]
        for lock in locks:
            await lock.acquire()
        try:
            yield
        finally:
            for lock in reversed(locks):
                lock.release()

    def locked(self, port: str) -> bool:
        lock = self._locks.get(port)
        return lock is not None and lock.locked()


# Singleton
port_locks = PortLockRegistry()
