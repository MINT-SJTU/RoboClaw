"""Hardware discovery orchestrator."""
from __future__ import annotations

import os
import sys
from dataclasses import replace

from roboclaw.embodied.embodiment.hardware.probers import _REGISTRY, get_prober
from roboclaw.embodied.embodiment.hardware.scan import (
    capture_camera_frames,
    scan_cameras,
    scan_serial_ports,
    suppress_stderr,
    restore_stderr,
)
from roboclaw.embodied.embodiment.interface.serial import SerialInterface
from roboclaw.embodied.embodiment.interface.video import VideoInterface


def _save_tty() -> list | None:
    """Save stdin terminal attributes (guard against pyserial corrupting them)."""
    try:
        import termios
        fd = sys.stdin.fileno()
        if os.isatty(fd):
            return termios.tcgetattr(fd)
    except (ImportError, OSError):
        pass
    return None


def _restore_tty(saved: list | None) -> None:
    """Restore stdin terminal attributes if they were saved."""
    if saved is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)
    except (ImportError, OSError):
        pass


class HardwareDiscovery:
    """Stateful hardware discovery workflow."""

    def __init__(self) -> None:
        self._scanned_ports: list[SerialInterface] = []
        self._scanned_cameras: list[VideoInterface] = []
        self._motion_active: bool = False

    @property
    def scanned_ports(self) -> list[SerialInterface]:
        return self._scanned_ports

    @property
    def scanned_cameras(self) -> list[VideoInterface]:
        return self._scanned_cameras

    @property
    def motion_active(self) -> bool:
        return self._motion_active

    def discover(self, model: str) -> list[SerialInterface]:
        """Probe ports using the protocol for the given model."""
        from roboclaw.embodied.embodiment.arm.registry import get_probe_config

        cfg = get_probe_config(model)
        prober = get_prober(cfg.protocol)
        ports = scan_serial_ports()
        result = self._probe_ports(
            ports, prober, cfg.protocol,
            motor_ids=list(cfg.motor_ids),
            baudrate=cfg.baudrate,
        )
        self._scanned_ports = result
        return result

    def discover_all(self) -> list[SerialInterface]:
        """Probe all ports with all registered probers (per-port, per-prober).

        Each port is tried with each prober independently, fixing the
        mixed-protocol bug where Koch arms were lost if SO101 was found first.
        """
        ports = scan_serial_ports()
        result: list[SerialInterface] = []
        for protocol, prober_cls in _REGISTRY.items():
            prober = prober_cls()
            matched = self._probe_ports(ports, prober, protocol)
            matched_devs = {p.dev for p in matched}
            result.extend(matched)
            ports = [p for p in ports if p.dev not in matched_devs]
            if not ports:
                break
        self._scanned_ports = result
        return result

    def discover_cameras(self) -> list[VideoInterface]:
        """Scan connected cameras."""
        result = scan_cameras()
        self._scanned_cameras = result
        return result

    def capture_camera_previews(self, output_dir: str) -> list[dict]:
        """Capture one preview frame per camera."""
        if not self._scanned_cameras:
            raise RuntimeError("No cameras scanned. Run discover_cameras first.")
        return capture_camera_frames(self._scanned_cameras, output_dir)

    def start_motion_detection(self) -> int:
        """Read baselines for all scanned ports. Returns port count."""
        if not self._scanned_ports:
            raise RuntimeError("No scanned ports. Run discover first.")
        saved = suppress_stderr()
        try:
            for port in self._scanned_ports:
                port.motion_detector.capture_baseline()
        finally:
            restore_stderr(saved)
        self._motion_active = True
        return len(self._scanned_ports)

    def poll_motion(self) -> list[dict]:
        """Read current positions and compute motion delta for each port."""
        if not self._motion_active:
            raise RuntimeError("Motion detection not started.")
        results = []
        saved = suppress_stderr()
        try:
            for port in self._scanned_ports:
                result = port.motion_detector.poll()
                results.append({
                    "port_id": port.stable_id,
                    "dev": port.dev,
                    "by_id": port.by_id,
                    "motor_ids": list(port.motor_ids),
                    "delta": result.delta,
                    "moved": result.moved,
                })
        finally:
            restore_stderr(saved)
        return results

    def stop_motion_detection(self) -> None:
        """Clear baselines and stop motion detection."""
        for port in self._scanned_ports:
            port.motion_detector.reset()
        self._motion_active = False

    def _probe_ports(
        self, ports: list[SerialInterface], prober, protocol: str = "",
        motor_ids: list[int] | None = None, baudrate: int = 1_000_000,
    ) -> list[SerialInterface]:
        """Probe ports with a single prober, handling permission errors."""
        from roboclaw.embodied.embodiment.hardware.scan import fix_serial_permissions

        saved = suppress_stderr()
        tty_saved = _save_tty()
        try:
            try:
                return self._do_probe(ports, prober, protocol, motor_ids=motor_ids, baudrate=baudrate)
            except Exception as exc:
                if "Permission denied" not in str(exc) and "Errno 13" not in str(exc):
                    raise
                if fix_serial_permissions():
                    return self._do_probe(ports, prober, protocol, motor_ids=motor_ids, baudrate=baudrate)
                raise PermissionError(
                    "Serial port permission denied. Run: bash scripts/setup-udev.sh"
                ) from exc
        finally:
            _restore_tty(tty_saved)
            restore_stderr(saved)

    @staticmethod
    def _do_probe(
        ports: list[SerialInterface], prober, protocol: str = "",
        motor_ids: list[int] | None = None, baudrate: int = 1_000_000,
    ) -> list[SerialInterface]:
        """Run the prober on each port, return those with motors."""
        result: list[SerialInterface] = []
        for port in ports:
            path = port.dev or port.by_id or port.by_path
            if not path:
                continue
            ids = prober.probe(path, baudrate=baudrate, motor_ids=motor_ids)
            if ids:
                result.append(replace(port, motor_ids=tuple(ids), bus_type=protocol))
        return result
