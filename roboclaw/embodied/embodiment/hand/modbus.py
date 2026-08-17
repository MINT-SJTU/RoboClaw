"""Shared Modbus RTU utilities for dexterous hand controllers."""

from __future__ import annotations

import struct


def crc16(data: bytes) -> int:
    """Modbus RTU CRC-16 (polynomial 0xA001, init 0xFFFF)."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def probe_modbus_slave_ids(
    port: str,
    baudrate: int,
    candidates: list[int] | range,
    register: int,
    register_count: int,
) -> list[int]:
    """Probe a serial port for responding Modbus RTU slave IDs.

    Sends a read-holding-registers (0x03) request to each candidate ID
    and returns those that respond with a valid frame.
    """
    import serial
    import time

    found: list[int] = []
    ser = serial.Serial(port, baudrate, timeout=0.2)
    try:
        for sid in candidates:
            frame = struct.pack(">BBHH", sid, 0x03, register, register_count)
            frame += struct.pack("<H", crc16(frame))
            ser.reset_input_buffer()
            ser.write(frame)
            time.sleep(0.1)
            resp = ser.read(5 + register_count * 2)
            if len(resp) >= 5 and resp[0] == sid and resp[1] == 0x03:
                found.append(sid)
    finally:
        ser.close()
    return found
