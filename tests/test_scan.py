from __future__ import annotations

from unittest.mock import patch

from roboclaw.embodied.embodiment.hardware.discovery import HardwareDiscovery
from roboclaw.embodied.embodiment.hardware.scan import (
    _list_serial_ports,
    scan_serial_ports,
    serial_patterns_for_platform,
)
from roboclaw.embodied.embodiment.interface.serial import SerialInterface


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
