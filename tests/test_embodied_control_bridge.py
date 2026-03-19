from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import pytest

from roboclaw.embodied.execution.integration.control_surfaces.ros2.control_surface import Ros2ControlSurfaceServer
from roboclaw.embodied.execution.integration.control_surfaces.ros2.scservo import ServoCalibration, probe_servo_register
from roboclaw.embodied.execution.integration.control_surfaces.ros2.so101_feetech import (
    ADDR_HOMING_OFFSET,
    ADDR_MAX_POSITION_LIMIT,
    ADDR_MIN_POSITION_LIMIT,
    So101FeetechRuntime,
)


def test_control_surface_server_rejects_unknown_robot_even_when_profile_matches() -> None:
    server = Ros2ControlSurfaceServer.__new__(Ros2ControlSurfaceServer)

    with pytest.raises(ValueError, match="Unknown control-surface ROS2 robot"):
        server._build_runtime(
            profile_id="so101_ros2_standard",
            robot_id="custom_arm",
            device_by_id="/dev/serial/by-id/usb-so101",
            calibration_path=None,
            calibration_id="so101_real",
        )


def test_control_surface_server_rejects_unknown_profile_even_when_robot_matches() -> None:
    server = Ros2ControlSurfaceServer.__new__(Ros2ControlSurfaceServer)

    with pytest.raises(ValueError, match="Unknown control-surface ROS2 profile"):
        server._build_runtime(
            profile_id="custom_profile",
            robot_id="so101",
            device_by_id="/dev/serial/by-id/usb-so101",
            calibration_path=None,
            calibration_id="so101_real",
        )


def test_so101_runtime_applies_calibration_before_motion() -> None:
    runtime = So101FeetechRuntime.__new__(So101FeetechRuntime)
    runtime._calibration = {
        "gripper": ServoCalibration(
            id=6,
            drive_mode=1,
            homing_offset=-120,
            range_min=100,
            range_max=500,
        )
    }
    writes: list[tuple[int, int, int]] = []
    runtime._write2 = lambda servo_id, address, data: writes.append((servo_id, address, data))

    runtime._apply_calibration("gripper")

    assert (6, ADDR_HOMING_OFFSET, So101FeetechRuntime._encode_signed_16(-120)) in writes
    assert (6, ADDR_MIN_POSITION_LIMIT, 100) in writes
    assert (6, ADDR_MAX_POSITION_LIMIT, 500) in writes


def test_so101_runtime_uses_drive_mode_in_gripper_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = So101FeetechRuntime.__new__(So101FeetechRuntime)
    runtime._calibration = {
        "gripper": ServoCalibration(
            id=6,
            drive_mode=1,
            homing_offset=0,
            range_min=100,
            range_max=500,
        )
    }

    assert runtime._normalized_to_raw("gripper", 0.0) == 500
    assert runtime._normalized_to_raw("gripper", 100.0) == 100

    monkeypatch.setattr(runtime, "read_gripper_position", lambda: 500)
    assert runtime.gripper_percent() == 0.0


def test_probe_servo_register_uses_open_port_once(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class FakePortHandler:
        def __init__(self, device: str) -> None:
            self.device = device
            self.baudrate = 0

        def openPort(self) -> bool:  # noqa: N802
            events.append(f"open:{self.baudrate}")
            return True

        def closePort(self) -> None:  # noqa: N802
            events.append("close")

        def getCurrentTime(self) -> float:  # noqa: N802
            return 0.0

    class FakePacketHandler:
        def __init__(self, protocol_version: int) -> None:
            self.protocol_version = protocol_version

        def read2ByteTxRx(self, port_handler: FakePortHandler, servo_id: int, address: int) -> tuple[int, int, int]:  # noqa: N802
            events.append(f"read:{servo_id}:{address}:{port_handler.device}")
            return 2048, 0, 0

        def getTxRxResult(self, result: int) -> str:  # noqa: N802
            return f"result={result}"

        def getRxPacketError(self, error: int) -> str:  # noqa: N802
            return f"error={error}"

    fake_sdk = types.SimpleNamespace(
        COMM_SUCCESS=0,
        PortHandler=FakePortHandler,
        PacketHandler=FakePacketHandler,
    )
    monkeypatch.setitem(sys.modules, "scservo_sdk", fake_sdk)

    result = probe_servo_register("/dev/serial/by-id/usb-so101", 6, 56, baudrate=1_000_000)

    assert result["ok"] is True
    assert events == [
        "open:1000000",
        "read:6:56:/dev/serial/by-id/usb-so101",
        "close",
    ]


def test_probe_servo_register_uses_resolved_host_device_for_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class FakePortHandler:
        def __init__(self, device: str) -> None:
            self.device = device
            self.baudrate = 0

        def openPort(self) -> bool:  # noqa: N802
            events.append(f"open:{self.baudrate}")
            return True

        def closePort(self) -> None:  # noqa: N802
            events.append("close")

        def getCurrentTime(self) -> float:  # noqa: N802
            return 0.0

    class FakePacketHandler:
        def __init__(self, protocol_version: int) -> None:
            self.protocol_version = protocol_version

        def read2ByteTxRx(self, port_handler: FakePortHandler, servo_id: int, address: int) -> tuple[int, int, int]:  # noqa: N802
            events.append(f"read:{servo_id}:{address}:{port_handler.device}")
            return 2048, 0, 0

        def getTxRxResult(self, result: int) -> str:  # noqa: N802
            return f"result={result}"

        def getRxPacketError(self, error: int) -> str:  # noqa: N802
            return f"error={error}"

    fake_sdk = types.SimpleNamespace(
        COMM_SUCCESS=0,
        PortHandler=FakePortHandler,
        PacketHandler=FakePacketHandler,
    )
    monkeypatch.setitem(sys.modules, "scservo_sdk", fake_sdk)
    monkeypatch.setattr(
        "roboclaw.embodied.execution.integration.control_surfaces.ros2.scservo.resolve_active_serial_device_path",
        lambda device: Path("/roboclaw-host-dev/ttyACM0"),
    )

    result = probe_servo_register("/dev/serial/by-id/usb-so101", 6, 56, baudrate=1_000_000)

    assert result["ok"] is True
    assert events == [
        "open:1000000",
        "read:6:56:/roboclaw-host-dev/ttyACM0",
        "close",
    ]


def test_control_surface_connect_auto_recovers_when_port_is_in_use() -> None:
    server = Ros2ControlSurfaceServer.__new__(Ros2ControlSurfaceServer)
    server._lock = threading.RLock()
    server._last_error = None
    server._last_result = {}

    calls: list[str] = []

    class FakeRuntime:
        def __init__(self) -> None:
            self.snapshot_calls = 0

        def connect(self) -> None:
            calls.append("connect")

        def disconnect(self) -> None:
            calls.append("disconnect")

        def snapshot(self) -> dict[str, object]:
            self.snapshot_calls += 1
            calls.append(f"snapshot:{self.snapshot_calls}")
            if self.snapshot_calls == 1:
                raise RuntimeError("read2(6, 56) failed: [TxRxResult] Port is in use!")
            return {"connected": True}

    server._runtime = FakeRuntime()

    response = types.SimpleNamespace(success=None, message=None)
    result = server._handle_connect(object(), response)

    assert result.success is True
    assert result.message == "connected"
    assert calls == ["connect", "snapshot:1", "disconnect", "connect", "connect", "snapshot:2"]


def test_control_surface_failure_translates_port_in_use_for_user() -> None:
    server = Ros2ControlSurfaceServer.__new__(Ros2ControlSurfaceServer)
    server._last_error = None
    server._last_result = {}

    response = types.SimpleNamespace(success=None, message=None)
    result = server._failure(response, RuntimeError("read2(6, 56) failed: [TxRxResult] Port is in use!"))

    assert result.success is False
    assert "serial port is still busy" in result.message
    assert "Port is in use" not in result.message
