"""Feetech (STS3215 / scservo_sdk) port prober."""
from __future__ import annotations

from roboclaw.embodied.embodiment.hardware.probers import register_prober

DEFAULT_BAUDRATE = 1_000_000
MOTOR_IDS = list(range(1, 7))

# STS3215 Present_Position register
_FEETECH_POS_ADDR = 56


class FeetechProber:
    """Probe and read Feetech servo motors on a serial port."""

    def probe(self, port_path: str, baudrate: int = DEFAULT_BAUDRATE, motor_ids: list[int] | None = None) -> list[int]:
        """Try reading Present_Position for Feetech motor IDs."""
        import scservo_sdk as scs

        ids = motor_ids or MOTOR_IDS
        handler = scs.PortHandler(port_path)
        try:
            if not handler.openPort():
                return []
        except OSError:
            return []
        try:
            handler.setBaudRate(baudrate)
            packet = scs.PacketHandler(0)
            found = []
            for mid in ids:
                val, result, _ = packet.read2ByteTxRx(handler, mid, _FEETECH_POS_ADDR)
                if result == scs.COMM_SUCCESS:
                    found.append(mid)
            return found
        finally:
            handler.closePort()

    def read_positions(
        self, port_path: str, motor_ids: list[int], baudrate: int = DEFAULT_BAUDRATE,
    ) -> dict[int, int]:
        """Read Feetech Present_Position for each motor ID."""
        import scservo_sdk as scs

        handler = scs.PortHandler(port_path)
        if not handler.openPort():
            return {}
        try:
            handler.setBaudRate(baudrate)
            packet = scs.PacketHandler(0)
            positions: dict[int, int] = {}
            for mid in motor_ids:
                val, result, _ = packet.read2ByteTxRx(handler, mid, _FEETECH_POS_ADDR)
                if result == scs.COMM_SUCCESS:
                    positions[mid] = val
            return positions
        finally:
            handler.closePort()


register_prober("feetech", FeetechProber)
