"""Minimal SCServo bus helpers for stage-1 ROS2 bridges."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


def _patch_set_packet_timeout(self, packet_length: int) -> None:  # noqa: N802
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50


def _decode_sign_magnitude(value: int, sign_bit: int) -> int:
    mask = (1 << sign_bit) - 1
    if value & (1 << sign_bit):
        return -(value & mask)
    return value & mask


@dataclass(frozen=True)
class ServoCalibration:
    """Calibration values for one servo."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int

    def clamp_raw(self, value: int) -> int:
        lower = min(self.range_min, self.range_max)
        upper = max(self.range_min, self.range_max)
        return max(lower, min(upper, value))

    def normalized_to_raw(self, percent: float) -> int:
        bounded = max(0.0, min(100.0, percent))
        ratio = bounded / 100.0
        if self.drive_mode:
            ratio = 1.0 - ratio
        raw = round(self.range_min + ((self.range_max - self.range_min) * ratio))
        return self.clamp_raw(raw)

    def raw_to_normalized(self, value: int) -> float:
        span = self.range_max - self.range_min
        if span == 0:
            return 0.0
        ratio = (value - self.range_min) / span
        if self.drive_mode:
            ratio = 1.0 - ratio
        return max(0.0, min(100.0, ratio * 100.0))


def load_calibration_file(path: str | Path) -> dict[str, ServoCalibration]:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return {
        name: ServoCalibration(
            id=int(item["id"]),
            drive_mode=int(item.get("drive_mode", 0)),
            homing_offset=int(item.get("homing_offset", 0)),
            range_min=int(item["range_min"]),
            range_max=int(item["range_max"]),
        )
        for name, item in data.items()
    }


class ScsServoBus:
    """Thin wrapper around `scservo_sdk` for position-mode commands."""

    TORQUE_ENABLE_ADDR = 40
    GOAL_POSITION_ADDR = 42
    LOCK_ADDR = 48
    PRESENT_POSITION_ADDR = 56
    P_COEFFICIENT_ADDR = 21
    D_COEFFICIENT_ADDR = 22
    I_COEFFICIENT_ADDR = 23
    OPERATING_MODE_ADDR = 33

    def __init__(self, device: str, *, baudrate: int = 1_000_000, protocol_version: int = 0) -> None:
        import scservo_sdk as scs

        self.device = device
        self.baudrate = baudrate
        self.protocol_version = protocol_version
        self._scs = scs
        self.port_handler = scs.PortHandler(device)
        self.port_handler.setPacketTimeout = _patch_set_packet_timeout.__get__(self.port_handler, scs.PortHandler)
        self.packet_handler = scs.PacketHandler(protocol_version)
        self.connected = False

    def connect(self) -> None:
        if self.connected:
            return
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open servo device `{self.device}`.")
        if not self.port_handler.setBaudRate(self.baudrate):
            self.port_handler.closePort()
            raise RuntimeError(f"Failed to set baudrate {self.baudrate} on `{self.device}`.")
        self.connected = True

    def disconnect(self) -> None:
        if not self.connected:
            return
        self.port_handler.closePort()
        self.connected = False

    def configure_position_mode(self, servo_id: int) -> None:
        self.write_byte(servo_id, self.OPERATING_MODE_ADDR, 0)
        self.write_byte(servo_id, self.P_COEFFICIENT_ADDR, 16)
        self.write_byte(servo_id, self.I_COEFFICIENT_ADDR, 0)
        self.write_byte(servo_id, self.D_COEFFICIENT_ADDR, 32)
        self.write_byte(servo_id, self.TORQUE_ENABLE_ADDR, 1)
        self.write_byte(servo_id, self.LOCK_ADDR, 1)

    def read_position(self, servo_id: int) -> int:
        value, result, error = self.packet_handler.read2ByteTxRx(
            self.port_handler,
            servo_id,
            self.PRESENT_POSITION_ADDR,
        )
        self._raise_if_failed(result, error, f"read present position for servo {servo_id}")
        return _decode_sign_magnitude(int(value), 15)

    def write_position(self, servo_id: int, raw_value: int) -> None:
        result, error = self.packet_handler.write2ByteTxRx(
            self.port_handler,
            servo_id,
            self.GOAL_POSITION_ADDR,
            int(raw_value),
        )
        self._raise_if_failed(result, error, f"write goal position for servo {servo_id}")

    def move_to_position(
        self,
        servo_id: int,
        raw_value: int,
        *,
        timeout_s: float = 2.5,
        tolerance_raw: int = 24,
        poll_s: float = 0.05,
    ) -> dict[str, float | int | bool]:
        target = int(raw_value)
        self.write_position(servo_id, target)
        deadline = time.time() + timeout_s
        last_position: int | None = None
        while time.time() < deadline:
            current = self.read_position(servo_id)
            last_position = current
            if abs(current - target) <= tolerance_raw:
                return {
                    "ok": True,
                    "target_raw": target,
                    "present_raw": current,
                    "delta_raw": current - target,
                }
            time.sleep(poll_s)
        return {
            "ok": False,
            "target_raw": target,
            "present_raw": last_position if last_position is not None else self.read_position(servo_id),
            "delta_raw": (last_position if last_position is not None else target) - target,
        }

    def write_byte(self, servo_id: int, address: int, value: int) -> None:
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler,
            servo_id,
            address,
            int(value),
        )
        self._raise_if_failed(result, error, f"write byte 0x{address:02x} for servo {servo_id}")

    def _raise_if_failed(self, result: int, error: int, operation: str) -> None:
        if result != self._scs.COMM_SUCCESS:
            raise RuntimeError(f"{operation} failed: {self.packet_handler.getTxRxResult(result)}")
        if error != 0:
            raise RuntimeError(f"{operation} failed: {self.packet_handler.getRxPacketError(error)}")


__all__ = [
    "ScsServoBus",
    "ServoCalibration",
    "load_calibration_file",
]
