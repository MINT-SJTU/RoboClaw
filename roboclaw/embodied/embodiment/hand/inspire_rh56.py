"""Inspire RH56 dexterous hand controller via Modbus RTU over serial."""

from __future__ import annotations

import struct
from contextlib import contextmanager

from roboclaw.embodied.embodiment.hand.modbus import crc16, probe_modbus_slave_ids
from roboclaw.embodied.embodiment.hand.registry import INSPIRE_RH56

# Modbus RTU register addresses (Inspire RH56 protocol)
_REG_ANGLE_SET = 1486   # write target angles (6 x uint16)
_REG_ANGLE_ACT = 1546   # read actual angles (6 x uint16)
_REG_FORCE_ACT = 1582   # read actual forces (6 x uint16)


def probe_slave_ids(port: str, candidates: range = range(1, 17)) -> list[int]:
    """Probe a serial port for responding Inspire hand Modbus slave IDs."""
    return probe_modbus_slave_ids(
        port, INSPIRE_RH56.baudrate, candidates,
        INSPIRE_RH56.probe_register, INSPIRE_RH56.probe_register_count,
    )


class _ModbusSerial:
    """Minimal Modbus RTU client for Inspire hand registers."""

    def __init__(self, port: str, slave_id: int, baudrate: int = INSPIRE_RH56.baudrate):
        import serial  # pyserial — already a roboclaw dependency

        self._ser = serial.Serial(port, baudrate, timeout=0.5)
        self._slave_id = slave_id

    def close(self) -> None:
        self._ser.close()

    def read_registers(self, addr: int, count: int) -> list[int]:
        """Read holding registers (function 0x03). Returns list of uint16 values."""
        import time

        frame = struct.pack(">BBHH", self._slave_id, 0x03, addr, count)
        frame += struct.pack("<H", crc16(frame))
        self._ser.reset_input_buffer()
        self._ser.write(frame)
        time.sleep(0.15)
        expected = 5 + count * 2  # slave + func + byte_count + data + crc
        resp = self._ser.read(expected)
        if len(resp) < expected:
            raise RuntimeError(f"Short response: expected {expected} bytes, got {len(resp)}")
        return list(struct.unpack(f">{count}H", resp[3 : 3 + count * 2]))

    def write_registers(self, addr: int, values: list[int]) -> None:
        """Write multiple registers (function 0x10)."""
        import time

        count = len(values)
        frame = struct.pack(">BBHHB", self._slave_id, 0x10, addr, count, count * 2)
        for v in values:
            frame += struct.pack(">H", v)
        frame += struct.pack("<H", crc16(frame))
        self._ser.reset_input_buffer()
        self._ser.write(frame)
        time.sleep(0.15)
        self._ser.read(8)  # ack frame: slave + func + addr(2) + count(2) + crc(2)


class InspireController:
    """Controls Inspire RH56 dexterous hand via Modbus RTU serial.

    Each method opens a serial connection, performs the operation, then closes it.
    Finger positions: [little, ring, middle, index, thumb_bend, thumb_rotation], range 0-1000.
    """

    def open_hand(self, port: str, slave_id: int = INSPIRE_RH56.default_slave_id) -> str:
        """Open all fingers to fully extended position."""
        with self._session(port, slave_id) as bus:
            bus.write_registers(_REG_ANGLE_SET, list(INSPIRE_RH56.open_positions))
        return "Hand opened."

    def close_hand(self, port: str, slave_id: int = INSPIRE_RH56.default_slave_id) -> str:
        """Close all fingers to gripped position."""
        with self._session(port, slave_id) as bus:
            bus.write_registers(_REG_ANGLE_SET, list(INSPIRE_RH56.close_positions))
        return "Hand closed."

    def set_pose(self, port: str, positions: list[int], slave_id: int = INSPIRE_RH56.default_slave_id) -> str:
        """Set individual finger positions (6 values, 0-1000)."""
        if len(positions) != INSPIRE_RH56.num_fingers:
            raise ValueError(f"Expected {INSPIRE_RH56.num_fingers} finger positions, got {len(positions)}.")
        if any(p < 0 or p > 1000 for p in positions):
            raise ValueError("Each finger position must be 0-1000.")
        with self._session(port, slave_id) as bus:
            bus.write_registers(_REG_ANGLE_SET, positions)
        summary = ", ".join(f"{label}={val}" for label, val in zip(INSPIRE_RH56.finger_labels, positions))
        return f"Pose set: {summary}."

    def get_status(self, port: str, slave_id: int = INSPIRE_RH56.default_slave_id) -> str:
        """Read current finger angles and forces."""
        with self._session(port, slave_id) as bus:
            angles = bus.read_registers(_REG_ANGLE_ACT, INSPIRE_RH56.num_fingers)
            forces = bus.read_registers(_REG_FORCE_ACT, INSPIRE_RH56.num_fingers)
        angle_dict = dict(zip(INSPIRE_RH56.finger_labels, angles))
        force_dict = dict(zip(INSPIRE_RH56.finger_labels, forces))
        return f"angles={angle_dict}\nforces={force_dict}"

    @staticmethod
    @contextmanager
    def _session(port: str, slave_id: int):
        """Open a Modbus RTU serial connection, yield it, and guarantee close."""
        bus = _ModbusSerial(port, slave_id)
        try:
            yield bus
        finally:
            bus.close()
