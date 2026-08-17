"""Dynamixel (XL430 / XL330) port prober."""
from __future__ import annotations

from roboclaw.embodied.embodiment.hardware.probers import register_prober

DEFAULT_BAUDRATE = 1_000_000
MOTOR_IDS = list(range(1, 7))

# Dynamixel Present_Position register
_DYNAMIXEL_POS_ADDR = 132


class DynamixelProber:
    """Probe and read Dynamixel servo motors on a serial port."""

    def probe(self, port_path: str, baudrate: int = DEFAULT_BAUDRATE, motor_ids: list[int] | None = None) -> list[int]:
        """Try reading Present_Position for Dynamixel motor IDs."""
        import dynamixel_sdk as dxl

        ids = motor_ids or MOTOR_IDS
        handler = dxl.PortHandler(port_path)
        try:
            if not handler.openPort():
                return []
        except OSError:
            return []
        try:
            handler.setBaudRate(baudrate)
            packet = dxl.PacketHandler(2.0)
            found = []
            for mid in ids:
                val, result, _ = packet.read4ByteTxRx(handler, mid, _DYNAMIXEL_POS_ADDR)
                if result == dxl.COMM_SUCCESS:
                    found.append(mid)
            return found
        finally:
            handler.closePort()

    def read_positions(
        self, port_path: str, motor_ids: list[int], baudrate: int = DEFAULT_BAUDRATE,
    ) -> dict[int, int]:
        """Read Dynamixel Present_Position for each motor ID."""
        import dynamixel_sdk as dxl

        handler = dxl.PortHandler(port_path)
        if not handler.openPort():
            return {}
        try:
            handler.setBaudRate(baudrate)
            packet = dxl.PacketHandler(2.0)
            positions: dict[int, int] = {}
            for mid in motor_ids:
                val, result, _ = packet.read4ByteTxRx(handler, mid, _DYNAMIXEL_POS_ADDR)
                if result == dxl.COMM_SUCCESS:
                    positions[mid] = val
            return positions
        finally:
            handler.closePort()


register_prober("dynamixel", DynamixelProber)
