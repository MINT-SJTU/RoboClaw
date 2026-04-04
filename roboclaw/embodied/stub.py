"""Hardware stubs for offline/CI testing of the embodied pipeline.

Set ``ROBOCLAW_STUB=1`` to activate.  All stub helpers live here
so that callers only need a single ``is_stub_mode()`` check.

Default stubs return two serial ports and one camera.  Override via
env vars for per-test flexibility:

    ROBOCLAW_STUB_PORTS      — JSON list of port dicts
    ROBOCLAW_STUB_CAMERAS    — JSON list of camera dicts
    ROBOCLAW_STUB_MOTORS     — JSON object mapping port by_id → motor id list
    ROBOCLAW_STUB_MOVED_PORT — by_id of the port that identify should detect
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

from roboclaw.embodied.interface.serial import SerialInterface
from roboclaw.embodied.interface.video import VideoInterface

_FLAG = "ROBOCLAW_STUB"

_DEFAULT_PORTS = [
    {
        "by_path": "/dev/serial/by-path/sim-pci-0:2.1",
        "by_id": "/dev/serial/by-id/usb-SIM_Serial_SIM001-if00",
        "dev": "/dev/ttyACM0",
    },
    {
        "by_path": "/dev/serial/by-path/sim-pci-0:2.2",
        "by_id": "/dev/serial/by-id/usb-SIM_Serial_SIM002-if00",
        "dev": "/dev/ttyACM1",
    },
]

_DEFAULT_CAMERAS = [
    {
        "by_path": "/dev/v4l/by-path/sim-cam0",
        "by_id": "usb-sim-cam0",
        "dev": "/dev/video0",
        "width": 640,
        "height": 480,
    },
]

_DEFAULT_MOTORS = {
    _DEFAULT_PORTS[0]["by_id"]: [1, 2, 3, 4, 5, 6],
    _DEFAULT_PORTS[1]["by_id"]: [1, 2, 3, 4, 5, 6],
}


def is_stub_mode() -> bool:
    """Return True when the stub environment variable is set."""
    return os.environ.get(_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


# ── Stub hardware data ───────────────────────────────────────────────


def stub_ports() -> list[SerialInterface]:
    """Return fake serial ports (overridable via ROBOCLAW_STUB_PORTS)."""
    raw = _read_json_env("ROBOCLAW_STUB_PORTS", _DEFAULT_PORTS)
    return [SerialInterface.from_dict(p) if isinstance(p, dict) else p for p in raw]


def stub_cameras() -> list[VideoInterface]:
    """Return fake cameras (overridable via ROBOCLAW_STUB_CAMERAS)."""
    raw = _read_json_env("ROBOCLAW_STUB_CAMERAS", _DEFAULT_CAMERAS)
    return [VideoInterface.from_dict(c) if isinstance(c, dict) else c for c in raw]


def stub_motors() -> dict[str, list[int]]:
    """Return motor ids keyed by port by_id (overridable via ROBOCLAW_STUB_MOTORS)."""
    return copy.deepcopy(_read_json_env("ROBOCLAW_STUB_MOTORS", _DEFAULT_MOTORS))


def stub_moved_port(ports: list[SerialInterface]) -> SerialInterface | None:
    """Return the port that identify should detect as 'moved'.

    Defaults to the first port, overridable via ROBOCLAW_STUB_MOVED_PORT.
    """
    target = os.environ.get("ROBOCLAW_STUB_MOVED_PORT", "")
    if target:
        for port in ports:
            if target in _port_paths(port):
                return port
    return ports[0] if ports else None


def stub_motor_ids(port_path: str) -> list[int]:
    """Resolve motor ids for a port path (matches across by_id/by_path/dev)."""
    motors = stub_motors()
    for port in stub_ports():
        if port_path not in _port_paths(port):
            continue
        for key in _port_paths(port):
            if key in motors:
                return list(motors[key])
        return []
    return list(motors.get(port_path, []))


# ── Internal helpers ─────────────────────────────────────────────────


def _port_paths(port: SerialInterface | dict) -> tuple[str, ...]:
    """Collect all meaningful path variants for one scanned port entry.

    Returns a tuple with fixed precedence: by_id, by_path, dev.
    Accepts both SerialInterface and legacy dict.
    """
    if isinstance(port, dict):
        return tuple(v for v in (
            port.get("by_id", ""),
            port.get("by_path", ""),
            port.get("dev", ""),
        ) if v)
    return tuple(v for v in (port.by_id, port.by_path, port.dev) if v)


def _read_json_env(name: str, default: Any) -> Any:
    """Read a JSON env var, falling back to *default*."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return json.loads(raw)
