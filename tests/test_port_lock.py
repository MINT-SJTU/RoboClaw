"""Tests for the cooperative serial port lock registry."""

import asyncio

import pytest

from roboclaw.embodied.port_lock import PortLockRegistry


@pytest.fixture
def registry() -> PortLockRegistry:
    return PortLockRegistry()


@pytest.mark.asyncio
async def test_acquire_same_port_blocks(registry: PortLockRegistry) -> None:
    """Acquiring the same port twice should block the second caller."""
    async with registry.acquire("/dev/ttyUSB0"):
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                registry.acquire("/dev/ttyUSB0").__aenter__(),
                timeout=0.05,
            )


@pytest.mark.asyncio
async def test_different_ports_independent(registry: PortLockRegistry) -> None:
    """Different ports should not block each other."""
    async with registry.acquire("/dev/ttyUSB0"):
        # This should complete immediately without blocking
        async with registry.acquire("/dev/ttyUSB1"):
            pass


@pytest.mark.asyncio
async def test_acquire_many(registry: PortLockRegistry) -> None:
    """acquire_many should hold all listed ports simultaneously."""
    ports = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2"]
    async with registry.acquire_many(ports):
        for port in ports:
            assert registry.locked(port), f"{port} should be locked"

    for port in ports:
        assert not registry.locked(port), f"{port} should be released"


@pytest.mark.asyncio
async def test_acquire_many_blocks_individual(registry: PortLockRegistry) -> None:
    """While acquire_many holds a port, single acquire should block."""
    ports = ["/dev/ttyUSB0", "/dev/ttyUSB1"]
    async with registry.acquire_many(ports):
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                registry.acquire("/dev/ttyUSB0").__aenter__(),
                timeout=0.05,
            )


@pytest.mark.asyncio
async def test_acquire_many_deduplicates(registry: PortLockRegistry) -> None:
    """Duplicate ports in acquire_many should not deadlock."""
    ports = ["/dev/ttyUSB0", "/dev/ttyUSB0", "/dev/ttyUSB1"]
    async with registry.acquire_many(ports):
        assert registry.locked("/dev/ttyUSB0")
        assert registry.locked("/dev/ttyUSB1")


@pytest.mark.asyncio
async def test_locked_reflects_state(registry: PortLockRegistry) -> None:
    """locked() should return True while held, False otherwise."""
    port = "/dev/ttyUSB0"
    assert not registry.locked(port)

    async with registry.acquire(port):
        assert registry.locked(port)

    assert not registry.locked(port)


@pytest.mark.asyncio
async def test_locked_unknown_port(registry: PortLockRegistry) -> None:
    """locked() on a never-used port returns False."""
    assert not registry.locked("/dev/ttyNONEXISTENT")


@pytest.mark.asyncio
async def test_release_after_acquire(registry: PortLockRegistry) -> None:
    """After exiting the context manager, the port should be acquirable again."""
    port = "/dev/ttyUSB0"
    async with registry.acquire(port):
        pass

    # Should succeed without blocking
    async with registry.acquire(port):
        assert registry.locked(port)
