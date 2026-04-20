from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from roboclaw.embodied.embodiment.hardware.discovery import HardwareDiscovery
from roboclaw.embodied.embodiment.hardware.scan import (
    _capture_camera_frame,
    _list_serial_ports,
    check_device_permissions,
    scan_serial_ports,
    scan_cameras,
    serial_patterns_for_platform,
)
from roboclaw.embodied.embodiment.interface.serial import SerialInterface
from roboclaw.embodied.embodiment.interface.video import VideoInterface


class _FakePort:
    def __init__(
        self,
        device: str,
        *,
        description: str = "n/a",
        hwid: str = "n/a",
        vid: int | None = None,
        pid: int | None = None,
    ) -> None:
        self.device = device
        self.description = description
        self.hwid = hwid
        self.vid = vid
        self.pid = pid


def test_list_serial_ports_uses_pyserial_devices_on_windows() -> None:
    with patch(
        "serial.tools.list_ports.comports",
        return_value=[
            _FakePort("/dev/cu.debug-console"),
            _FakePort("/dev/cu.usbmodemB", description="USB Serial", hwid="USB VID:PID=1A86:55D3", vid=0x1A86, pid=0x55D3),
            _FakePort("/dev/cu.usbmodemA", description="USB Serial", hwid="USB VID:PID=1A86:55D3", vid=0x1A86, pid=0x55D3),
        ],
    ), patch("roboclaw.embodied.embodiment.hardware.scan.os.name", "nt"):
        ports = _list_serial_ports()

    assert ports == ["/dev/cu.debug-console", "/dev/cu.usbmodemA", "/dev/cu.usbmodemB"]


def test_list_serial_ports_matches_lerobot_range_on_linux() -> None:
    class _FakePath:
        def __init__(self, value: str) -> None:
            self._value = value

        def __str__(self) -> str:
            return self._value

    with patch(
        "pathlib.Path.glob",
        return_value=[_FakePath("/dev/ttyACM0"), _FakePath("/dev/ttyUSB1")],
    ), patch("roboclaw.embodied.embodiment.hardware.scan.os.name", "posix"), patch(
        "roboclaw.embodied.embodiment.hardware.scan.sys.platform", "linux",
    ):
        ports = _list_serial_ports()

    assert ports == ["/dev/ttyACM0", "/dev/ttyUSB1"]


def test_scan_serial_ports_merges_port_list_with_linux_symlink_aliases() -> None:
    with (
        patch("roboclaw.embodied.embodiment.hardware.scan._list_serial_ports", return_value=["/dev/ttyACM0"]),
        patch(
            "roboclaw.embodied.embodiment.hardware.scan._read_symlink_map",
            side_effect=[
                {"/dev/ttyACM0": "/dev/serial/by-path/pci-0:2.1"},
                {"/dev/ttyACM0": "/dev/serial/by-id/usb-ABC-if00"},
            ],
        ),
        patch("roboclaw.embodied.embodiment.hardware.scan.os.path.exists", return_value=True),
    ):
        ports = scan_serial_ports()

    assert ports == [
        SerialInterface(by_path="/dev/serial/by-path/pci-0:2.1", by_id="/dev/serial/by-id/usb-ABC-if00", dev="/dev/ttyACM0"),
    ]


def test_scan_serial_ports_uses_lerobot_compatible_range() -> None:
    with (
        patch("roboclaw.embodied.embodiment.hardware.scan._read_symlink_map", return_value={}),
        patch("roboclaw.embodied.embodiment.hardware.scan.os.path.exists", return_value=True),
        patch("roboclaw.embodied.embodiment.hardware.scan._list_serial_ports", return_value=["/dev/cu.usbmodemA"]),
    ):
        ports = scan_serial_ports()

    assert ports == [SerialInterface(dev="/dev/cu.usbmodemA")]


def test_serial_patterns_for_platform_macos_uses_cu_only() -> None:
    with patch("roboclaw.embodied.embodiment.hardware.scan.sys.platform", "darwin"):
        patterns = serial_patterns_for_platform()
    assert all(p.startswith("cu.") for p in patterns)
    assert not any(p.startswith("tty.") for p in patterns)


def test_serial_patterns_for_platform_linux_uses_tty() -> None:
    with patch("roboclaw.embodied.embodiment.hardware.scan.sys.platform", "linux"):
        patterns = serial_patterns_for_platform()
    assert all(p.startswith("tty") for p in patterns)


def test_scan_serial_ports_macos_only_returns_cu_devices() -> None:
    with (
        patch(
            "roboclaw.embodied.embodiment.hardware.scan._list_serial_ports",
            return_value=["/dev/cu.usbmodem123"],
        ),
        patch("roboclaw.embodied.embodiment.hardware.scan._read_symlink_map", return_value={}),
        patch("roboclaw.embodied.embodiment.hardware.scan.os.path.exists", return_value=True),
    ):
        ports = scan_serial_ports()

    assert len(ports) == 1
    assert ports[0].dev == "/dev/cu.usbmodem123"


def test_discovery_probes_directly_on_scanned_port() -> None:
    class _FakeProber:
        def probe(self, port_path: str, baudrate: int = 1_000_000, motor_ids: list[int] | None = None) -> list[int]:
            return [1, 2, 3] if port_path == "/dev/cu.usbmodem123" else []

    ports = [SerialInterface(dev="/dev/cu.usbmodem123")]
    result = HardwareDiscovery._do_probe(ports, _FakeProber(), "feetech")

    assert result == [SerialInterface(dev="/dev/cu.usbmodem123", bus_type="feetech", motor_ids=(1, 2, 3))]


class _FakeCapture:
    def __init__(self, index: int, opened: bool = True) -> None:
        self.index = index
        self._opened = opened

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop: int) -> float:
        values = {
            3: 1280.0,
            4: 720.0,
            5: 30.0,
        }
        return values.get(prop, 0.0)

    def release(self) -> None:
        return None


class _FakeCv2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_AVFOUNDATION = 1200

    def __init__(self, opened: set[int]) -> None:
        self._opened = opened

    def VideoCapture(self, source, backend=None):  # noqa: N802
        index = source if isinstance(source, int) else int(source)
        return _FakeCapture(index=index, opened=index in self._opened)


def test_scan_cameras_macos_probes_indices() -> None:
    fake_cv2 = _FakeCv2(opened={0, 2})

    with (
        patch("roboclaw.embodied.embodiment.hardware.scan.sys.platform", "darwin"),
        patch(
            "roboclaw.embodied.embodiment.hardware.scan._list_macos_camera_inventory",
            return_value=[
                {"index": 0, "name": "MacBook Neo相机", "unique_id": "builtin-camera"},
                {"index": 2, "name": "USB相机", "unique_id": "usb-camera"},
            ],
        ),
        patch("roboclaw.embodied.embodiment.hardware.scan.suppress_stderr", return_value=1),
        patch("roboclaw.embodied.embodiment.hardware.scan.restore_stderr"),
        patch.dict(sys.modules, {"cv2": fake_cv2}),
    ):
        cameras = scan_cameras()

    assert cameras == [
        VideoInterface(by_path="MacBook Neo相机", by_id="builtin-camera", dev="0", width=1280, height=720, fps=30),
        VideoInterface(by_path="USB相机", by_id="usb-camera", dev="2", width=1280, height=720, fps=30),
    ]


def test_check_device_permissions_macos_reports_camera_access_issue() -> None:
    with (
        patch("roboclaw.embodied.embodiment.hardware.scan.sys.platform", "darwin"),
        patch(
            "roboclaw.embodied.embodiment.hardware.scan._list_macos_camera_inventory",
            return_value=[{"index": 1, "name": "USB相机", "unique_id": "usb-camera"}],
        ),
        patch("roboclaw.embodied.embodiment.hardware.scan.scan_cameras", return_value=[]),
    ):
        permissions = check_device_permissions()

    assert permissions == {
        "serial": {"ok": True, "count": 0},
        "camera": {"ok": False, "count": 1},
        "platform": "darwin",
        "hint": "Grant camera access to Terminal or RoboClaw, then retry scanning.",
    }


def test_capture_camera_frame_macos_uses_stable_id_preview_source(tmp_path: Path) -> None:
    camera = VideoInterface(
        by_path="USB相机",
        by_id="usb-camera",
        dev="2",
        width=1280,
        height=720,
        fps=30,
    )
    preview = {
        "image_path": str(tmp_path / "00-preview.jpg"),
        "preview_key": "00-preview",
        "width": 1280,
        "height": 720,
    }

    with (
        patch("roboclaw.embodied.embodiment.hardware.scan.sys.platform", "darwin"),
        patch(
            "roboclaw.embodied.embodiment.hardware.scan._capture_camera_frame_macos",
            return_value=preview,
        ) as capture_macos,
    ):
        result = _capture_camera_frame(object(), camera, tmp_path, "00-preview")

    assert result == preview
    capture_macos.assert_called_once()
