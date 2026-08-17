"""BrainCo Revo2 dexterous hand controller via bc_stark_sdk (Modbus RS-485)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from roboclaw.embodied.embodiment.hand.modbus import probe_modbus_slave_ids
from roboclaw.embodied.embodiment.hand.registry import REVO2

_SPEEDS = [1000] * REVO2.num_fingers


def probe_slave_ids(port: str, candidates: list[int] | None = None) -> list[int]:
    """Probe port for responding Revo2 Modbus slave IDs."""
    cands = candidates or list(REVO2.probe_candidates or range(1, 17))
    return probe_modbus_slave_ids(
        port, REVO2.baudrate, cands,
        register=REVO2.probe_register, register_count=REVO2.probe_register_count,
    )


class Revo2Controller:
    """Controls BrainCo Revo2 dexterous hand via bc_stark_sdk.

    Each method opens a connection, performs the operation, then closes it.
    Finger positions: [thumb, thumb_aux, index, middle, ring, pinky], 0-1000.
    """

    async def open_hand(self, port: str, slave_id: int = REVO2.default_slave_id) -> str:
        """Open all fingers."""
        async with self._session(port, slave_id) as client:
            await client.set_finger_positions_and_speeds(slave_id, list(REVO2.open_positions), _SPEEDS)
        return "Hand opened."

    async def close_hand(self, port: str, slave_id: int = REVO2.default_slave_id) -> str:
        """Close all fingers."""
        async with self._session(port, slave_id) as client:
            await client.set_finger_positions_and_speeds(slave_id, list(REVO2.close_positions), _SPEEDS)
        return "Hand closed."

    async def set_pose(self, port: str, positions: list[int], slave_id: int = REVO2.default_slave_id) -> str:
        """Set individual finger positions (6 values, 0-1000)."""
        if len(positions) != REVO2.num_fingers:
            raise ValueError(f"Expected {REVO2.num_fingers} finger positions, got {len(positions)}.")
        if any(p < 0 or p > 1000 for p in positions):
            raise ValueError("Each finger position must be 0-1000.")
        async with self._session(port, slave_id) as client:
            await client.set_finger_positions_and_speeds(slave_id, positions, _SPEEDS)
        summary = ", ".join(f"{label}={val}" for label, val in zip(REVO2.finger_labels, positions))
        return f"Pose set: {summary}."

    async def get_status(self, port: str, slave_id: int = REVO2.default_slave_id) -> str:
        """Read current finger positions, speeds, and currents."""
        async with self._session(port, slave_id) as client:
            status = await client.get_motor_status(slave_id)
        pos = dict(zip(REVO2.finger_labels, status.positions))
        spd = dict(zip(REVO2.finger_labels, status.speeds))
        cur = dict(zip(REVO2.finger_labels, status.currents))
        return f"positions={pos}\nspeeds={spd}\ncurrents={cur}"

    @staticmethod
    @asynccontextmanager
    async def _session(port: str, slave_id: int):
        """Open bc_stark_sdk connection, yield client, guarantee close."""
        from bc_stark_sdk import main_mod as libstark  # lazy import

        client = await libstark.modbus_open(port, libstark.Baudrate.Baud460800)
        if not client:
            raise RuntimeError("Failed to open hand serial connection.")
        info = await client.get_device_info(slave_id)
        if not info:
            libstark.modbus_close(client)
            raise RuntimeError("Hand not responding. Check connection and power.")
        await client.set_finger_unit_mode(slave_id, libstark.FingerUnitMode.Normalized)
        try:
            yield client
        finally:
            libstark.modbus_close(client)
